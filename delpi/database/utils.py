import re

import polars as pl
import numpy as np

from delpi.chem.modification import Modification
from delpi.chem.modification_param import (
    MOD_SEPARATOR,
    get_unimod_id,
    get_mod_loc_index,
)
from delpi.chem.modification_param import (
    PROT_C_TERM_AA,
    PROT_N_TERM_AA,
    PEPT_C_TERM_AA,
    PEPT_N_TERM_AA,
    MOD_SEPARATOR,
)

TERM_CHARS = [PROT_N_TERM_AA, PROT_C_TERM_AA, PEPT_N_TERM_AA, PEPT_C_TERM_AA]

max_mod_id = Modification.get_max_accession_num()
mod_name_array = [None] * (max_mod_id + 1)
for mod in Modification.name_to_mod_map.values():
    mod_name_array[mod.accession_num] = mod.name
mod_name_array = np.asarray(mod_name_array, dtype=np.str_)


def get_modified_sequence(
    seq: str, mod_ids: str, mod_sites: str, use_unimod_id: bool = True
) -> str:
    """Create a peptide sequence string with modifications

    Args:
        seq (str): a plain peptide sequence with terminal residues
        mod_ids (str): modification param indexes separated by `;`
        mod_sites (str): modification sites separated by `;`
        use_unimod_id (bool, optional): a flag to apply UniMod ID. Defaults to True.

    Returns:
        str: modified sequence string
    """
    if mod_ids is None or len(mod_ids) < 1:
        return seq

    seq = list(seq)
    mod_ids = map(int, mod_ids.split(MOD_SEPARATOR))
    mod_sites = map(int, mod_sites.split(MOD_SEPARATOR))

    for mod_id, pos in zip(mod_ids, mod_sites):
        unimod_id = get_unimod_id(mod_id)
        pos = len(seq) - 1 if pos == -1 else pos
        # mod_loc_idx = get_mod_loc_index(mod_id)
        # if mod_loc_idx == 0:
        #     pos += 1
        # elif mod_loc_idx % 2 == 1:  # N-term
        #     pos = 0
        # else:  # C-term
        #     pos = len(seq) - 1
        seq[pos] += (
            f"(UniMod:{unimod_id})"
            if use_unimod_id
            else f"({mod_name_array[unimod_id]})"
        )
    return "".join(seq)


def create_peptidoform_df(
    peptide_df: pl.DataFrame,
    modification_df: pl.DataFrame,
    modified_sequence_format: str = "delpi",
):

    mod_df = modification_df
    modified_sequence_format = modified_sequence_format.lower()
    # [TODO] support other formats (Skyline, etc)
    if modified_sequence_format not in ["diann", "skyline", "unimod", "delpi"]:
        raise NotImplementedError()

    use_unimod_id = modified_sequence_format in ["diann", "delpi"]

    peptidoform_df = mod_df.with_columns(peptide_df["peptide"][mod_df["peptide_index"]])

    peptidoform_df = peptidoform_df.select(
        pl.col("peptide_index", "peptidoform_index"),
        pl.when(pl.col("mod_ids").is_null())
        .then(pl.col("peptide"))
        .otherwise(
            pl.struct(["peptide", "mod_ids", "mod_sites"]).map_elements(
                lambda x: get_modified_sequence(
                    x["peptide"],
                    x["mod_ids"],
                    x["mod_sites"],
                    use_unimod_id=use_unimod_id,
                ),
                return_dtype=pl.String,
            )
        )
        .alias("modified_sequence"),
    )

    # adjust terminal chars
    if modified_sequence_format != "delpi":
        term_ptrns = f"[{''.join([re.escape(c) for c in TERM_CHARS])}]"
        peptidoform_df = peptidoform_df.with_columns(
            pl.col("modified_sequence").str.replace_all(term_ptrns, "")
        )
    else:
        peptidoform_df = peptidoform_df.with_columns(
            pl.col("modified_sequence")
            .str.replace_all(PROT_N_TERM_AA, PEPT_N_TERM_AA)
            .str.replace_all(PROT_C_TERM_AA, PEPT_C_TERM_AA)
        )

    return peptidoform_df
