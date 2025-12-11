import polars as pl

from delpi import DATA_DIR
from delpi.constants import C13C12_MASS_DIFF


averagine_df = pl.read_parquet(DATA_DIR / "averagine.parquet").with_columns(
    pl.col("mass").cast(pl.UInt32).alias("nominal_mass"), pl.col("envelope")
)


def get_precursor_lib_df(max_isotopes, min_mass=300, max_mass=7000):

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
            pl.int_ranges(pl.col("envelope").list.len(), dtype=pl.UInt8).alias(
                "isotope_index"
            ),
        )
        .explode(["isotope_index", "envelope", "rank"])
        .filter(pl.col("isotope_index") < max_isotopes)
        .rename({"envelope": "predicted_intensity"})
        .sort(["mass", "rank"])
        .with_columns(get_offset_mass, get_normalized_intensity)
    )

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
