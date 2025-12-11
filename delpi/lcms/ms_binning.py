import polars as pl


class MassSpecBinning:
    """
    This class group mass values with bin widths
    that increase proportionally to the mass.
    The bin width can be specified in the unit of parts-per-million (ppm).
    Measurement errors in mass-spec are proportional to the mass.
    """

    def __init__(self, min_mz, max_mz, bin_width_in_ppm):

        assert min_mz + 5 < max_mz, "max_mz should be greater than min_mz + 5"
        self.bin_width_in_ppm = bin_width_in_ppm

        bin_edges = list()
        bin_end = min_mz - 5
        while bin_end < max_mz + 5:
            bin_edges.append(bin_end)
            bin_end += bin_end * bin_width_in_ppm * 1e-6
        bin_edges.append(bin_end)
        self.df = pl.DataFrame(
            bin_edges, schema={"bin_edges": pl.Float32}
        ).with_row_index(name="bin_index")

        self.min_mz = min_mz
        self.max_mz = max_mz

    @property
    def bin_start(self):
        return self.df.item(0, "bin_edges")

    @property
    def bin_end(self):
        return self.df.item(-1, "bin_edges")

    def get_bin_index(self, mz_values):
        bin_indices = self.df["bin_edges"].search_sorted(mz_values).rename("bin_index")

        return bin_indices

    def get_range(self, bin_indices):
        assert isinstance(bin_indices, pl.Series)
        return (
            bin_indices.to_frame(name="bin_index")
            .with_columns((pl.col("bin_index") - 1).alias("bin_st_idx"))
            .join(self.df, left_on="bin_st_idx", right_on="bin_index", how="left")
            .rename({"bin_edges": "min"})
            .join(self.df, left_on="bin_index", right_on="bin_index", how="left")
            .rename({"bin_edges": "max"})
            .drop("bin_st_idx")
        )

    def get_min(self, bin_indices):
        return (
            bin_indices.to_frame(name="bin_index")
            .with_columns((pl.col("bin_index") - 1).alias("bin_st_idx"))
            .join(self.df, left_on="bin_st_idx", right_on="bin_index", how="left")
            .rename({"bin_edges": "min"})
            .drop("bin_st_idx")
        )

    def get_max(self, bin_indices):
        return (
            bin_indices.to_frame(name="bin_index")
            .join(self.df, left_on="bin_index", right_on="bin_index", how="left")
            .rename({"bin_edges": "max"})
        )

    def get_center(self, bin_indices):
        return self.get_range(bin_indices).select(
            pl.col("bin_index"), (0.5 * (pl.col("min") + pl.col("max"))).alias("center")
        )
