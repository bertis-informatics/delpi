from typing import Sequence, Tuple, Union, Dict, Any, Self, List
from pathlib import Path

import polars as pl
import torch

from delpi.lcms.base_ion_type import BaseIonType
from delpi.chem.amino_acid import AminoAcid
from delpi.chem.modification_param import ModificationParam
from delpi.database.fasta_parser import FastaParser
from delpi.database.enzyme import Enzyme
from delpi.database.decoy_generator import DecoyGenerator
from delpi.database.modification_handler import ModificationHandler
from delpi.database.precursor_generator import PrecursorGenerator
from delpi.database.spec_lib_generator import SpectralLibGenerator
from delpi.database.utils import create_peptidoform_df
from delpi.utils.yaml_file import load_yaml, save_yaml

PEPTIDE_SEQ_REGEX = rf"^[{AminoAcid.peptide_n_term}{AminoAcid.protein_n_term}]?[{AminoAcid.standard_amino_acid_chars}]+[{AminoAcid.peptide_c_term}{AminoAcid.protein_c_term}]?$"


class PeptideDatabase:

    def __init__(self):

        self.params = dict()
        self.sequence_df = None
        self.peptide_df = None
        self.modification_df = None
        self.precursor_df = None
        self.speclib_df = None

        self.prefix_mass_container = None

    @staticmethod
    def extract_params(config_dict):
        param_dict = dict()
        for key in ["fasta_file", "decoy"]:
            if key in config_dict:
                param_dict[key] = config_dict[key]

        for group_key in ["digest", "modification", "precursor", "fragment"]:
            if group_key in config_dict:
                param_dict.update(config_dict[group_key])
        return param_dict

    def build(
        self,
        fasta_file: str,
        enzyme: str = "trypsin",
        min_len: int = 7,
        max_len: int = 30,
        max_missed_cleavages: int = 1,
        decoy: str = None,
        n_term_methionine_excision: bool = True,
        mod_param_set: Union[
            Sequence[ModificationParam],
            Sequence[Tuple[str, str, str, bool]],
            Sequence[Dict[str, Any]],
        ] = None,
        max_mods: int = 2,
        min_precursor_charge: int = 1,
        max_precursor_charge: int = 4,
        min_fragment_charge: int = 1,
        max_fragment_charge: int = 2,
        min_precursor_mz: float = 300,
        max_precursor_mz: float = 1800,
        min_fragment_mz: float = 200,
        max_fragment_mz: float = 1800,
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        max_fragments=16,
        device: Union[str, torch.device] = "cuda:0",
        use_multiprocessing: bool = False,
        *args,
        **kwargs,
    ) -> Self:

        # read and parse FASTA file
        parser = FastaParser(fasta_file)
        sequence_df = parser.parse().with_row_index("protein_index")

        # digest FASTA sequences
        enzyme_ = Enzyme(
            name=enzyme,
            min_len=min_len,
            max_len=max_len,
            n_term_methionine_excision=n_term_methionine_excision,
            max_missed_cleavages=max_missed_cleavages,
        )

        # digest protein sequences
        peptide_df = enzyme_.digest(
            sequence_df, use_multiprocessing=use_multiprocessing
        )

        # filter peptides containing only standard amino acids
        peptide_df = peptide_df.filter(
            pl.col("peptide").str.contains(PEPTIDE_SEQ_REGEX)
        )

        # generate decoy peptides
        decoy_generator = DecoyGenerator(method=decoy)
        peptide_df = decoy_generator.append_decoys(peptide_df)
        peptide_df = peptide_df.with_row_index("peptide_index")

        # apply modifications
        mod_handler = ModificationHandler(mod_param_set, max_mods=max_mods)
        modification_df = mod_handler.apply(
            peptide_df, use_multiprocessing=use_multiprocessing
        )
        modification_df = modification_df.with_row_index("peptidoform_index")

        # generate precursors
        precursor_generator = PrecursorGenerator(
            min_charge=min_precursor_charge,
            max_charge=max_precursor_charge,
            min_mz=min_precursor_mz,
            max_mz=max_precursor_mz,
        )
        precursor_df, modification_df, prefix_mass_container = (
            precursor_generator.generate_precursors(peptide_df, modification_df)
        )
        precursor_df = precursor_df.with_row_index("precursor_index")

        # generate spectral library
        speclib_generator = SpectralLibGenerator(
            min_charge=min_fragment_charge,
            max_charge=max_fragment_charge,
            prefix_ion_type=prefix_ion_type,
            suffix_ion_type=suffix_ion_type,
            max_fragments=max_fragments,
            apply_phospho=mod_handler.has_phospho,
            device=device,
        )
        speclib_df, modification_df = speclib_generator.generate_spectral_lib(
            peptide_df,
            modification_df,
            precursor_df,
            prefix_mass_container,
            min_fragment_mz=min_fragment_mz,
            max_fragment_mz=max_fragment_mz,
        )

        self.params.update(
            {
                "fasta_file": str(parser.fasta_path),
                "digest": enzyme_.param_dict,
                "decoy": decoy_generator.method,
                "modification": mod_handler.param_dict,
                "precursor": precursor_generator.param_dict,
                "fragment": speclib_generator.param_dict,
            }
        )
        self.sequence_df = sequence_df
        self.peptide_df = peptide_df
        self.modification_df = modification_df
        self.precursor_df = precursor_df
        self.speclib_df = speclib_df
        self.prefix_mass_container = prefix_mass_container

        return self

    def get_prefix_mass_array(self, peptidoform_index: int):
        return self.prefix_mass_container.get_prefix_mass_array(peptidoform_index)

    def find_precursors(
        self, min_precursor_mz: float, max_precursor_mz: float
    ) -> pl.DataFrame:

        if self.precursor_df is None:
            raise ValueError("precursor_df is not generated yet")

        precursor_df = self.precursor_df
        ret_df = precursor_df.filter(
            pl.col("precursor_mz").is_between(min_precursor_mz, max_precursor_mz)
        )

        if isinstance(ret_df, pl.LazyFrame):
            ret_df = ret_df.collect()

        return ret_df.with_columns(
            self.modification_df["ref_rt"][ret_df["peptidoform_index"]]
        )

    def get_peptidoform_df(self, modified_sequence_format: str = "delpi"):

        return create_peptidoform_df(
            self.peptide_df,
            self.modification_df,
            modified_sequence_format=modified_sequence_format,
        )

    @classmethod
    def load(cls, save_dir: str):

        save_dir = Path(save_dir)
        param_file_path = save_dir / "param.yaml"
        # hdf_file_path = save_dir / 'data.hdf'

        param_dict = load_yaml(param_file_path)
        new_instance = cls()

        new_instance.params.update(param_dict)
        for param_key, param_val in new_instance.__dict__.items():
            if param_key.endswith("_df"):
                fpath = Path(rf"{save_dir}\{param_key}.parquet")
                if fpath.exists():
                    param_val = pl.read_parquet(fpath)
                    setattr(new_instance, param_key, param_val)

        # if hdf_file_path.exists():
        #     with h5py.File(hdf_file_path, 'r') as hf:
        #         for param_key, param_val in hf.items():
        #             setattr(new_instance, param_key, np.array(param_val))

        return new_instance

    def check_db_files(self, save_dir) -> bool:
        save_dir = Path(save_dir)
        param_file_path = save_dir / "param.yaml"
        ret = dict()
        ret[param_file_path] = param_file_path.exists()
        for param_key, param_val in self.__dict__.items():
            if isinstance(param_val, pl.DataFrame):
                fpath = save_dir / f"{param_key}.parquet"
                ret[fpath] = fpath.exists()
        return ret

    @classmethod
    def exists(cls, save_dir) -> bool:
        check_dict = cls().check_db_files(save_dir)
        return all(check_dict.values())

    def save(self, save_dir: str, overwrite: bool = False) -> Path:

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        param_file_path = save_dir / "param.yaml"

        if not overwrite:
            file_check_dict = self.check_db_files(save_dir)
            if any(file_check_dict.values()):
                raise FileExistsError()

        for param_key, param_val in self.__dict__.items():
            if isinstance(param_val, pl.DataFrame):
                fpath = save_dir / f"{param_key}.parquet"
                param_val.write_parquet(fpath)

        save_yaml(param_file_path, self.params)

    @staticmethod
    def join(
        db_dir: Path,
        pmsm_df: pl.DataFrame,
        precursor_columns: List[str] = [
            "precursor_charge",
            "precursor_mz",
        ],
        modification_columns: List[str] = [
            "mod_ids",
            "mods",
            "mod_sites",
            "precursor_mass",
            "ref_rt",
        ],
        peptide_columns: List[str] = [
            "peptide",
            "sequence_length",
            "is_decoy",
            # "protein_index",
        ],
    ) -> pl.DataFrame:
        """Helpers to join pmsm_df with database tables.

        Args:
            db_dir (Path): _description_
            pmsm_df (pl.DataFrame): _description_
            precursor_columns (List[str], optional): _description_. Defaults to [ "precursor_charge", "precursor_mz", ].
            modification_columns (List[str], optional): _description_. Defaults to [ "mod_ids", "mods", "mod_sites", "precursor_mass", "ref_rt", ].
            peptide_columns (List[str], optional): _description_. Defaults to [ "peptide", "sequence_length", "is_decoy", ].

        Returns:
            pl.DataFrame: _description_
        """

        precursor_df = pl.scan_parquet(db_dir / "precursor_df.parquet")
        peptide_df = pl.scan_parquet(db_dir / "peptide_df.parquet")
        modification_df = pl.scan_parquet(db_dir / "modification_df.parquet")

        label_df = (
            pmsm_df.lazy()
            .join(
                precursor_df.select(
                    pl.col("peptidoform_index", "precursor_index", *precursor_columns)
                ),
                on="precursor_index",
                how="left",
            )
            .join(
                modification_df.select(
                    pl.col("peptidoform_index", "peptide_index", *modification_columns)
                ),
                on="peptidoform_index",
                how="left",
            )
            .join(
                peptide_df.select(pl.col("peptide_index", *peptide_columns)),
                on="peptide_index",
                how="left",
            )
        ).collect()

        return label_df
