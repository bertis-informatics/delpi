import polars as pl

from delpi import DATA_DIR
from delpi.constants import C13C12_MASS_DIFF, PRECURSOR_ISOTOPE_INDICES

averagine_df = pl.read_parquet(DATA_DIR / "averagine.parquet").with_columns(
    pl.col("mass").cast(pl.UInt32).alias("nominal_mass"), pl.col("envelope")
)


def get_precursor_lib_df(max_isotopes, min_mass=300, max_mass=7000):
    """Averagine isotopes PRECURSOR_ISOTOPE_INDICES[:max_isotopes] per mass.

    Each mass has max_isotopes rows: the non-negative isotopes in
    predicted-intensity rank order, then M-1 (predicted intensity 0), if
    selected, last.
    """
    isotopes = PRECURSOR_ISOTOPE_INDICES[:max_isotopes]
    if len(isotopes) != max_isotopes or max_isotopes < 1:
        raise ValueError(
            f"max_isotopes must be in [1, {len(PRECURSOR_ISOTOPE_INDICES)}]"
        )
    max_isotope_index = max(isotopes)

    get_offset_mass = (
        (pl.col("isotope_index") * C13C12_MASS_DIFF)
        .cast(pl.Float32)
        .alias("offset_mass")
    )

    get_normalized_intensity = pl.col("predicted_intensity") / pl.col(
        "predicted_intensity"
    ).max().over("mass")

    avg_df = (
        averagine_df.filter(pl.col("mass").is_between(min_mass, max_mass))
        .with_columns(
            pl.col("envelope")
            .list.eval(pl.element().rank(descending=True).cast(pl.UInt8))
            .alias("rank"),
            pl.int_ranges(pl.col("envelope").list.len(), dtype=pl.Int8).alias(
                "isotope_index"
            ),
        )
        .explode(["isotope_index", "envelope", "rank"], empty_as_null=True)
        .filter(pl.col("isotope_index") <= max_isotope_index)
        .rename({"envelope": "predicted_intensity"})
    )
    if -1 in isotopes:
        m1_df = avg_df.unique("mass").with_columns(
            pl.lit(-1, dtype=pl.Int8).alias("isotope_index"),
            pl.lit(0.0, dtype=avg_df.schema["predicted_intensity"]).alias(
                "predicted_intensity"
            ),
            pl.lit(255, dtype=pl.UInt8).alias("rank"),
        )
        avg_df = pl.concat([avg_df, m1_df.select(avg_df.columns)])

    avg_df = avg_df.sort(
        ["mass", pl.col("isotope_index") < 0, "rank"]
    ).with_columns(get_offset_mass, get_normalized_intensity)

    return avg_df


# df = get_precursor_lib_df(3, min_mass=50, max_mass=5000)
# df1 = df.group_by(pl.col('mass')).agg(pl.all().last())

# from matplotlib import pyplot as plt
# z = np.polyfit(df1['mass'], df1['predicted_intensity'], 8)
# p = np.poly1d(z)


# plt.figure()
# plt.plot(df1['mass'], df1['predicted_intensity'])
# plt.plot(df1['mass'], p(df1['mass']))
# plt.savefig('./temp/averagine.jpg')
