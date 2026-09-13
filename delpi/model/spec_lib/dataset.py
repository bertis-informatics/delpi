import numpy as np
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

        # Every table's primary key is 0..N-1 and row-position-aligned, so
        # foreign keys resolve via plain array indexing instead of a join:
        # precursor.peptidoform_index -> modification_df row
        #   -> .peptide_index -> peptide_df row.
        self.peptide_index_by_peptidoform_arr = modification_df[
            "peptide_index"
        ].to_numpy()
        sequence_length_by_peptide = peptide_df["sequence_length"].to_numpy()
        self.mod_ids_arr = modification_df["mod_ids"].to_numpy()
        self.mod_sites_arr = modification_df["mod_sites"].to_numpy()
        self.peptide_seq_arr = peptide_df["peptide"].to_numpy()

        if self.is_precursor_level:
            self.precursor_index_arr = precursor_df["precursor_index"].to_numpy()
            self.precursor_charge_arr = precursor_df["precursor_charge"].to_numpy()
            peptidoform_index_arr = precursor_df["peptidoform_index"].to_numpy()
        else:
            peptidoform_index_arr = np.arange(modification_df.shape[0], dtype=np.uint32)

        self.peptidoform_index_arr = peptidoform_index_arr
        # one-time array for batch grouping only; not kept per-item (that's
        # re-derived from peptide_index_by_peptidoform_arr in __getitem__).
        sequence_length_arr = sequence_length_by_peptide[
            self.peptide_index_by_peptidoform_arr[peptidoform_index_arr]
        ]

        # only what SeqDataBatchSampler needs for grouping/counting batches.
        self.label_df = pl.DataFrame({"sequence_length": sequence_length_arr})

    @property
    def labels(self):
        return self.label_df

    def __len__(self):
        return self.peptidoform_index_arr.shape[0]

    def __getitem__(self, index):

        sample = dict()
        peptidoform_index = int(self.peptidoform_index_arr[index])
        peptide_index = int(self.peptide_index_by_peptidoform_arr[peptidoform_index])

        if self.is_precursor_level:
            precursor_charge = self.precursor_charge_arr[index]
            x_meta = torch.FloatTensor(
                [
                    precursor_charge,
                    DEFAULT_NCE,
                    DEFAULT_FRAGMENTATION,
                    DEFAULT_MASS_ANALYZER,
                ]
            )

            sample["precursor_index"] = int(self.precursor_index_arr[index])
            sample["x_meta"] = x_meta
        else:
            sample["peptidoform_index"] = peptidoform_index

        mod_ids = self.mod_ids_arr[peptidoform_index]
        mod_sites = self.mod_sites_arr[peptidoform_index]
        seq_str = self.peptide_seq_arr[peptide_index]

        x_aa = encode_sequence(seq_str)
        x_mod = encode_modification_feature_from_strings(
            mod_sites, mod_ids, x_aa.shape[0]
        )

        sample["x_aa"] = x_aa
        sample["x_mod"] = x_mod

        return sample
