from typing import List, Tuple
import numpy as np
import numba as nb

from delpi.chem.amino_acid import AminoAcid
from delpi.chem.modification import Modification
from delpi.chem.modification_param import MOD_SEPARATOR, get_unimod_id


MAX_MOD_COUNT = 16

AA_CHARS = sorted(list(AminoAcid.standard_amino_acid_chars))
AA_INDEX_MAP = np.full(128, dtype=np.int8, fill_value=-1)

# Standard amino acids
for aa_index, aa in enumerate(AA_CHARS):
    AA_INDEX_MAP[ord(aa)] = aa_index

# Terminal residues
for aa in [AminoAcid.peptide_n_term.residue, AminoAcid.protein_n_term.residue]:
    AA_INDEX_MAP[ord(aa)] = len(AA_CHARS)
for aa in [AminoAcid.peptide_c_term.residue, AminoAcid.protein_c_term.residue]:
    AA_INDEX_MAP[ord(aa)] = len(AA_CHARS) + 1


ATOM_SYMBOLS = [
    "H",
    "C",
    "N",
    "O",
    "P",
    "S",
    "2H",
    "18O",
    "F",
    "Na",
    "Se",
    "Li",
    "13C",
    "15N",
]

ATOM_INDEX_MAP = {atom_code: i for i, atom_code in enumerate(ATOM_SYMBOLS)}
max_unimod_id = max(mod.accession_num for mod in Modification.name_to_mod_map.values())
MOD_FEATURE_MAP = np.zeros([max_unimod_id + 1, len(ATOM_INDEX_MAP)], dtype=np.float32)
for mod in Modification.name_to_mod_map.values():
    for atom_code, c in mod.composition.get_elements().items():
        atom_idx = ATOM_INDEX_MAP.get(atom_code, -1)
        if atom_idx >= 0:
            MOD_FEATURE_MAP[mod.accession_num, atom_idx] = c


@nb.njit(nogil=True, cache=True, fastmath=True)
def encode_sequence(seq_str: str, out_arr: np.ndarray = None):
    if out_arr is None:
        out_arr = np.zeros(len(seq_str), dtype=np.int8)
    for i, a in enumerate(seq_str):
        out_arr[i] = AA_INDEX_MAP[ord(a)]
    return out_arr


def encode_batch_sequences(batch_sequences: List[str]) -> np.ndarray:
    seq_len = len(batch_sequences[0])
    x_aa = np.zeros((len(batch_sequences), seq_len), dtype=np.int8)
    for i, seq_str in enumerate(batch_sequences):
        _ = encode_sequence(seq_str, x_aa[i])
    return x_aa


@nb.njit(nogil=True, cache=True, fastmath=True)
def parse_mod_string(s):
    result = []
    current = 0
    for i in range(len(s)):
        c = s[i]
        if c == MOD_SEPARATOR:
            result.append(current)
            current = 0
        else:
            current = current * 10 + (ord(c) - ord("0"))
    result.append(current)
    return np.array(result, dtype=np.int32)


@nb.njit(nogil=True, cache=True, fastmath=True)
def encode_modification(mod_sites: str, mod_ids: str, out_arr: np.ndarray = None):
    """parse modification strings and encode to a fixed-size array"""

    if out_arr is None:
        out_arr = np.full((MAX_MOD_COUNT, 2), -1, dtype=np.int16)

    if mod_sites is None or mod_ids is None or len(mod_sites) == 0 or len(mod_ids) == 0:
        out_arr[:] = -1
        return out_arr

    mod_site_arr = parse_mod_string(mod_sites)
    mod_id_arr = parse_mod_string(mod_ids)

    for i in range(len(mod_site_arr)):
        site = mod_site_arr[i]
        # unimod_accession_num = get_unimod_id(mod_id_arr[i])
        unimod_accession_num = (mod_id_arr[i] >> 16) & 0xFFFF
        out_arr[i, 0] = site
        out_arr[i, 1] = unimod_accession_num

    return out_arr


@nb.njit(nogil=True, cache=True, fastmath=True)
def encode_modification_feature_from_strings(
    mod_sites: str, mod_ids: str, seq_len: int
):
    x_mod_feature = np.zeros((seq_len, MOD_FEATURE_MAP.shape[-1]), dtype=np.float32)
    if mod_sites is None or mod_ids is None or len(mod_sites) == 0 or len(mod_ids) == 0:
        return x_mod_feature

    mod_site_arr = parse_mod_string(mod_sites)
    mod_id_arr = parse_mod_string(mod_ids)
    for i in range(len(mod_site_arr)):
        site = mod_site_arr[i]
        unimod_accession_num = (mod_id_arr[i] >> 16) & 0xFFFF
        x_mod_feature[site, :] += MOD_FEATURE_MAP[unimod_accession_num]

    return x_mod_feature


@nb.njit(nogil=True, cache=True, fastmath=True)
def encode_modification_feature(x_mod: np.ndarray, seq_len: int):
    x_mod_feature = np.zeros((seq_len, MOD_FEATURE_MAP.shape[-1]), dtype=np.float32)
    for j in range(x_mod.shape[0]):
        site = x_mod[j, 0]
        unimod_accession_num = x_mod[j, 1]
        if unimod_accession_num < 0:  # accession_num < 0 means no modification
            break
        x_mod_feature[site, :] += MOD_FEATURE_MAP[unimod_accession_num]
    return x_mod_feature


def encode_modification_tuples(
    modification_tuples: List[Tuple[int, int]], out_arr: np.ndarray = None
):
    # modification_tuples = [(0, 737), (0, 1), (2, 737), (4, 35), (5, 737), (5, 121), (15, 2)]
    if out_arr is None:
        out_arr = np.full((MAX_MOD_COUNT, 2), -1, dtype=np.int16)

    for i, (site, unimod_accession_num) in enumerate(modification_tuples):
        out_arr[i, :] = site, unimod_accession_num

    return out_arr


def encode_modification_strings(
    mod_sites: str, mod_ids: str, out_arr: np.ndarray = None
):
    if mod_sites is None or mod_ids is None:
        return encode_modification_tuples([], out_arr)

    modification_tuples = [
        (int(site), get_unimod_id(int(mod_id)))
        for mod_id, site in zip(
            mod_ids.split(MOD_SEPARATOR), mod_sites.split(MOD_SEPARATOR)
        )
    ]

    return encode_modification_tuples(modification_tuples, out_arr)


# def encode_modification_feature_from_strings(
#     mod_sites: str, mod_ids: str, seq_len: int
# ):
#     if mod_sites is None or mod_ids is None:
#         return np.zeros((seq_len, MOD_FEATURE_MAP.shape[-1]), dtype=np.float32)

#     x_mod = np.array(
#         [
#             (int(site), get_unimod_id(int(mod_id)))
#             for mod_id, site in zip(
#                 mod_ids.split(MOD_SEPARATOR), mod_sites.split(MOD_SEPARATOR)
#             )
#         ],
#         dtype=np.int16,
#     )

#     return encode_modification_feature(x_mod, seq_len)
