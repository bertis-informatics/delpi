"""Batch encoding of peptidoforms or precursors into predictor inputs.

Produces the same x_aa/x_mod/x_meta as PeptideDataset with default collate,
but parses sequences and modification strings once into flat arrays and
fills whole batches with one njit call, so inference loops skip per-item
Python work.
"""

import numba as nb
import numpy as np
import polars as pl

from delpi.chem.modification_param import MOD_SEPARATOR
from delpi.constants import DEFAULT_NCE
from delpi.model.spec_lib.aa_encoder import AA_INDEX_MAP, MOD_FEATURE_MAP
from delpi.model.spec_lib.dataset import DEFAULT_FRAGMENTATION, DEFAULT_MASS_ANALYZER

MOD_FEATURE_DIM = MOD_FEATURE_MAP.shape[-1]


@nb.njit(parallel=True, nogil=True, cache=True)
def _encode_batch(
    peptidoform_idx,
    peptide_of_peptidoform,
    seq_offsets,
    seq_codes,
    mod_offsets,
    mod_sites,
    mod_unimod_ids,
    mod_feature_map,
    x_aa,
    x_mod,
):
    """Fill x_aa[:n, :L] and x_mod[:n, :L]; return L, or -1 if token lengths differ."""
    n = peptidoform_idx.shape[0]
    p0 = peptide_of_peptidoform[peptidoform_idx[0]]
    L = seq_offsets[p0 + 1] - seq_offsets[p0]
    for k in range(n):
        p = peptide_of_peptidoform[peptidoform_idx[k]]
        if seq_offsets[p + 1] - seq_offsets[p] != L:
            return -1
    for k in nb.prange(n):
        pf = peptidoform_idx[k]
        s0 = seq_offsets[peptide_of_peptidoform[pf]]
        for j in range(L):
            x_aa[k, j] = seq_codes[s0 + j]
        x_mod[k, :L, :] = 0.0
        for m in range(mod_offsets[pf], mod_offsets[pf + 1]):
            # -1 is the C-terminal token (ModificationHandler)
            site = mod_sites[m] + L if mod_sites[m] < 0 else mod_sites[m]
            x_mod[k, site, :] += mod_feature_map[mod_unimod_ids[m]]
    return L


def _parse_mod_column(modification_df: pl.DataFrame, column: str) -> pl.Series:
    # null or "" means no modification; "" becomes null before the cast
    col = pl.col(column)
    return modification_df.select(
        pl.when(col.str.len_bytes() > 0)
        .then(col)
        .str.split(MOD_SEPARATOR)
        .cast(pl.List(pl.Int64))
        .fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
        .alias(column)
    ).to_series()


class PeptideBatchEncoder:
    """Encode batches of items, which are precursor_df rows when precursor_df
    is given and modification_df rows otherwise.

    Tables are indexed by row position (primary key == row), like
    PeptideDataset. ``labels`` holds the per-item sequence_length used by
    SeqDataBatchSampler/ChunkedSeqDataBatchSampler.
    """

    def __init__(
        self,
        peptide_df: pl.DataFrame,
        modification_df: pl.DataFrame,
        precursor_df: pl.DataFrame = None,
        nce: float = DEFAULT_NCE,
    ):
        self.nce = nce

        peptides = peptide_df["peptide"]
        token_len = peptides.str.len_bytes().to_numpy().astype(np.int64)
        self.seq_offsets = np.zeros(token_len.shape[0] + 1, dtype=np.int64)
        np.cumsum(token_len, out=self.seq_offsets[1:])
        seq_bytes = np.frombuffer("".join(peptides.to_list()).encode("ascii"), np.uint8)
        if seq_bytes.shape[0] != self.seq_offsets[-1]:
            raise ValueError("peptide sequences must be ASCII")
        self.seq_codes = AA_INDEX_MAP[seq_bytes]
        self.max_token_len = int(token_len.max()) if token_len.shape[0] else 0

        mod_ids = _parse_mod_column(modification_df, "mod_ids")
        mod_sites = _parse_mod_column(modification_df, "mod_sites")
        mod_counts = mod_ids.list.len().to_numpy()
        if not np.array_equal(mod_counts, mod_sites.list.len().to_numpy()):
            raise ValueError("mod_ids and mod_sites have different lengths")
        self.mod_offsets = np.zeros(mod_counts.shape[0] + 1, dtype=np.int64)
        np.cumsum(mod_counts, out=self.mod_offsets[1:])
        self.mod_unimod_ids = (
            (mod_ids.explode().drop_nulls().to_numpy() >> 16) & 0xFFFF
        ).astype(np.int64)
        self.mod_sites = mod_sites.explode().drop_nulls().to_numpy().astype(np.int64)
        if self.mod_unimod_ids.shape[0] and (
            self.mod_unimod_ids.max() >= MOD_FEATURE_MAP.shape[0]
        ):
            raise ValueError("modification UniMod id outside MOD_FEATURE_MAP")

        self.peptide_of_peptidoform = (
            modification_df["peptide_index"].to_numpy().astype(np.int64)
        )
        mod_token_len = token_len[
            np.repeat(self.peptide_of_peptidoform, mod_counts.astype(np.int64))
        ]
        if np.any(
            (self.mod_sites >= mod_token_len) | (self.mod_sites < -mod_token_len)
        ):
            raise ValueError("modification site outside the peptide tokens")

        if precursor_df is None:
            self.peptidoform_index_arr = np.arange(
                modification_df.height, dtype=np.int64
            )
            self.precursor_charge_arr = None
        else:
            self.peptidoform_index_arr = (
                precursor_df["peptidoform_index"].to_numpy().astype(np.int64)
            )
            self.precursor_charge_arr = (
                precursor_df["precursor_charge"].to_numpy().astype(np.float32)
            )

        sequence_length = peptide_df["sequence_length"].to_numpy()
        self.label_df = pl.DataFrame(
            {
                "sequence_length": sequence_length[
                    self.peptide_of_peptidoform[self.peptidoform_index_arr]
                ]
            }
        )

    @property
    def labels(self) -> pl.DataFrame:
        return self.label_df

    def __len__(self):
        return self.peptidoform_index_arr.shape[0]

    def token_length(self, index: int) -> int:
        """Token count (residues + 2 terminal tokens) of item ``index``."""
        p = self.peptide_of_peptidoform[self.peptidoform_index_arr[index]]
        return int(self.seq_offsets[p + 1] - self.seq_offsets[p])

    def encode(self, index_arr, x_aa, x_mod, x_meta=None) -> int:
        """Fill x_aa[:n, :L], x_mod[:n, :L] and x_meta[:n] for items index_arr.

        Every item must have the same token length L (batches are grouped by
        sequence_length); returns L. x_aa is an integer array of shape
        (>= n, >= L), x_mod float32 (>= n, >= L, MOD_FEATURE_DIM), x_meta
        float32 (>= n, 4) and filled only for precursor items.
        """
        index_arr = np.asarray(index_arr, dtype=np.int64)
        n = index_arr.shape[0]
        if n == 0:
            raise ValueError("empty batch")
        peptidoform_idx = self.peptidoform_index_arr[index_arr]
        L = _encode_batch(
            peptidoform_idx,
            self.peptide_of_peptidoform,
            self.seq_offsets,
            self.seq_codes,
            self.mod_offsets,
            self.mod_sites,
            self.mod_unimod_ids,
            MOD_FEATURE_MAP,
            x_aa,
            x_mod,
        )
        if L < 0:
            raise ValueError("batch items have different token lengths")
        if x_meta is not None:
            if self.precursor_charge_arr is None:
                raise ValueError("x_meta needs precursor items")
            x_meta[:n, 0] = self.precursor_charge_arr[index_arr]
            x_meta[:n, 1] = self.nce
            x_meta[:n, 2] = DEFAULT_FRAGMENTATION
            x_meta[:n, 3] = DEFAULT_MASS_ANALYZER
        return int(L)
