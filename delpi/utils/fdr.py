import polars as pl
import numpy as np


def fdr_to_q_values(fdr_values: np.ndarray) -> np.ndarray:
    """Converts FDR values to q-values.
    The q-value can be understood as the minimal FDR at which a PSM is accepted
    Takes a ascending sorted array of FDR values and converts them to q-values.
    for every element the lowest FDR where it would be accepted is used as q-value.
    """
    fdr_values_flipped = np.flip(fdr_values)
    q_values_flipped = np.minimum.accumulate(fdr_values_flipped)
    q_vals = np.flip(q_values_flipped)
    return q_vals


def calculate_q_value(
    tda_df: pl.DataFrame,
    score_column: str = "score",
    target_to_decoy_size_ratio: int = 1,
    pi_zero: int = 1,
    out_column: str = "q_value",
    score_sort_descending=True,
):

    count_decoys = pl.col("is_decoy").cum_sum().alias("n_decoys")
    count_targets = (
        pl.int_range(1, pl.len() + 1, dtype=pl.UInt32) - pl.col("n_decoys")
    ).alias("n_targets")
    calc_fdr = (
        (
            (pl.col("n_decoys") * target_to_decoy_size_ratio * pi_zero)
            / pl.col("n_targets")
        )
        .cast(pl.Float32)
        .alias("fdr")
    )

    tda_df = tda_df.with_row_index("_index")

    sorted_tda_df = tda_df.select(pl.col("_index", score_column, "is_decoy")).sort(
        [score_column], descending=score_sort_descending
    )

    fdr_arr = (
        sorted_tda_df.with_columns(count_decoys)
        .with_columns(count_targets)
        .select(calc_fdr)["fdr"]
        .to_numpy()
    )
    q_value_arr = fdr_to_q_values(fdr_arr)
    sorted_tda_df = sorted_tda_df.with_columns(
        pl.Series(name=out_column, values=q_value_arr, dtype=pl.Float32)
    )

    return (
        tda_df.select(pl.exclude(out_column))
        .join(
            sorted_tda_df.select(pl.col("_index", out_column)),
            on="_index",
            how="left",
        )
        .drop("_index")
    )


def test_fdr():
    df = pl.DataFrame(
        {"is_decoy": [1, 1, 0, 1, 0, 0], "score": [1, 2, 3, 4, 5, 6]},
        schema={"is_decoy": pl.Int8, "score": pl.Float32},
    ).with_columns(pl.col("is_decoy").cast(pl.Boolean))

    tda_df = calculate_q_value(df, score_column="score")
