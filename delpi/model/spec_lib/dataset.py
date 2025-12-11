import torch
import polars as pl
from torch.utils.data import Dataset

from delpi.model.spec_lib.aa_encoder import (
    encode_sequence,
    encode_modification_feature_from_strings,
)


DEFAULT_NCE = 30
DEFAULT_FRAGMENTATION = 0
DEFAULT_MASS_ANALYZER = 0


class PeptideDataset(Dataset):

    def __init__(
        self,
        precursor_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        peptide_df: pl.DataFrame,
        level: str = "precursor",  # "precursor" or "peptidoform"
    ):
        assert level in ["precursor", "peptidoform"], "Invalid dataset level"

        self.is_precursor_level = level == "precursor"
        if self.is_precursor_level:
            label_df = (
                precursor_df.select(pl.col("precursor_index", "peptidoform_index"))
                .join(
                    modification_df.select(
                        pl.col("peptidoform_index", "peptide_index")
                    ),
                    on="peptidoform_index",
                    how="left",
                )
                .join(
                    peptide_df.select(pl.col("peptide_index", "sequence_length")),
                    on="peptide_index",
                    how="left",
                )
            )
        else:
            label_df = modification_df.select(
                pl.col("peptidoform_index", "peptide_index")
            ).join(
                peptide_df.select(pl.col("peptide_index", "sequence_length")),
                on="peptide_index",
                how="left",
            )

        self.precursor_df = precursor_df
        self.modification_df = modification_df
        self.peptide_df = peptide_df
        self.label_df = label_df

    def __len__(self):
        return self.label_df.shape[0]

    def __getitem__(self, index):

        label_df = self.label_df
        precursor_df = self.precursor_df
        modification_df = self.modification_df
        peptide_df = self.peptide_df
        sample = dict()

        # index = 3
        if self.is_precursor_level:
            precursor_index = label_df.item(index, "precursor_index")
            peptidoform_index = label_df.item(index, "peptidoform_index")
            peptide_index = label_df.item(index, "peptide_index")
            precursor_charge = precursor_df.item(index, "precursor_charge")
            x_meta = torch.FloatTensor(
                [
                    precursor_charge,
                    DEFAULT_NCE,
                    DEFAULT_FRAGMENTATION,
                    DEFAULT_MASS_ANALYZER,
                ]
            )

            sample["precursor_index"] = precursor_index
            sample["x_meta"] = x_meta
        else:
            peptidoform_index = index
            peptide_index = label_df.item(index, "peptide_index")
            sample["peptidoform_index"] = peptidoform_index

        mod_ids = modification_df.item(peptidoform_index, "mod_ids")
        mod_sites = modification_df.item(peptidoform_index, "mod_sites")
        seq_str = peptide_df.item(peptide_index, "peptide")

        x_aa = encode_sequence(seq_str)
        x_mod = encode_modification_feature_from_strings(
            mod_sites, mod_ids, x_aa.shape[0]
        )

        sample["x_aa"] = x_aa
        sample["x_mod"] = x_mod

        return sample


# class PmsmDataset(Dataset):

#     def __init__(
#         self,
#         pmsm_df: pl.DataFrame,
#         nce: float = DEFAULT_NCE,
#         fragmentation: int = DEFAULT_FRAGMENTATION,
#         mass_analyzer: int = DEFAULT_MASS_ANALYZER,
#     ):
#         self.label_df = pmsm_df
#         self.nce = nce
#         self.fragmentation = fragmentation
#         self.mass_analyzer = mass_analyzer

#     def __len__(self):
#         return self.label_df.shape[0]

#     def __getitem__(self, index):

#         pmsm_df = self.label_df
#         precursor_index = pmsm_df.item(index, "precursor_index")
#         precursor_charge = pmsm_df.item(index, "precursor_charge")
#         mod_ids = pmsm_df.item(index, "mod_ids")
#         mod_sites = pmsm_df.item(index, "mod_sites")
#         seq_str = pmsm_df.item(index, "peptide")

#         x_meta = torch.FloatTensor(
#             [
#                 precursor_charge,
#                 self.nce,
#                 self.fragmentation,
#                 self.mass_analyzer,
#             ]
#         )

#         x_aa = encode_sequence(seq_str)
#         x_mod = encode_modification_feature_from_strings(
#             mod_sites, mod_ids, x_aa.shape[0]
#         )

#         return {
#             "precursor_index": precursor_index,
#             "x_aa": x_aa,
#             "x_mod": x_mod,
#             "x_meta": x_meta,
#         }
