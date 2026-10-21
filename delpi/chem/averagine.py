import polars as pl

from delpi import DATA_DIR
from delpi.constants import C13C12_MASS_DIFF

averagine_df = pl.read_parquet(DATA_DIR / "averagine.parquet").with_columns(
    pl.col("mass").cast(pl.UInt32).alias("nominal_mass"), pl.col("envelope")
)


def get_isotope_indices(max_isotopes):
    """Total isotope count, starting at M-1 and always including M."""
    if not 2 <= max_isotopes <= 8:
        raise ValueError("max_isotopes must be between 2 and 8")
    return list(range(-1, max_isotopes - 1))


def get_precursor_lib_df(max_isotopes, min_mass=300, max_mass=7000):
    isotopes = pl.DataFrame(
        {"isotope_index": get_isotope_indices(max_isotopes)},
        schema={"isotope_index": pl.Int8},
    )
    return (
        averagine_df.filter(pl.col("mass").is_between(min_mass, max_mass))
        .join(isotopes, how="cross")
        .with_columns(
            pl.when(pl.col("isotope_index") < 0)
            .then(0.0)
            .otherwise(
                pl.col("envelope")
                .list.get(pl.col("isotope_index").clip(lower_bound=0), null_on_oob=True)
                .fill_null(0.0)
            )
            .cast(pl.Float32)
            .alias("predicted_intensity")
        )
        .with_columns(
            (pl.col("isotope_index") * C13C12_MASS_DIFF)
            .cast(pl.Float32)
            .alias("offset_mass"),
            (
                pl.col("predicted_intensity")
                / pl.col("predicted_intensity").max().over("mass")
            ).alias("predicted_intensity"),
            pl.col("predicted_intensity")
            .rank("ordinal", descending=True)
            .over("mass")
            .cast(pl.UInt8)
            .alias("rank"),
        )
        .drop("envelope")
        .sort("mass", "isotope_index")
    )
