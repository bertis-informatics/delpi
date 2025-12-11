import re
from typing import Dict, List, Any
from itertools import combinations
from typing import Sequence
import multiprocessing as mp
from functools import partial
from typing import Sequence, Tuple, Union

import polars as pl
import numpy as np
import pandas as pd

from delpi.chem.modification_param import (
    ModificationParam,
    MOD_SEPARATOR,
)


class ModificationHandler:

    def __init__(
        self,
        mod_param_set: Union[
            Sequence[ModificationParam],
            Sequence[Tuple[str, str, str, bool]],
            Sequence[Dict[str, Any]],
        ],
        max_mods: int = 1,
    ):

        if mod_param_set is None:
            mod_param_set = []

        if len(mod_param_set) > 0:
            if isinstance(mod_param_set[0], Tuple):
                mod_param_set = [ModificationParam(*p) for p in mod_param_set]
            elif isinstance(mod_param_set[0], Dict):
                mod_param_set = [ModificationParam(**p) for p in mod_param_set]
            elif isinstance(mod_param_set[0], ModificationParam):
                pass
            else:
                raise NotImplementedError()

        self.mod_param_set = mod_param_set
        self.max_mods = max_mods

    @property
    def param_dict(self):
        return {
            "max_mods": self.max_mods,
            "mod_param_set": [p.to_dict() for p in self.mod_param_set],
        }

    @property
    def has_phospho(self):
        return np.any(
            [mod.modification.name == "Phospho" for mod in self.mod_param_set]
        )

    @staticmethod
    def generate_static_mods(
        seq: str, static_mods: List[ModificationParam], id_to_name_map: Dict[int, str]
    ):

        # get_site = lambda site : min(max(site - 1, 0), len(seq) - 3)
        get_site = lambda site: -1 if site == len(seq) - 1 else site
        fixed_mod_ids, fixed_mods, fixed_mod_sites = [], [], []
        for mod in static_mods:
            for match in re.finditer(mod.get_re_pattern(), seq):
                fixed_mod_ids.append(mod.id)
                fixed_mods.append(id_to_name_map[mod.id])
                fixed_mod_sites.append(get_site(match.start()))
                # fixed_mod_sites.append(min(max(match.start() - 1, 0), len(seq) - 3))
        return (
            MOD_SEPARATOR.join(map(str, fixed_mod_ids)),
            MOD_SEPARATOR.join(fixed_mods),
            MOD_SEPARATOR.join(map(str, fixed_mod_sites)),
        )

    @staticmethod
    def generate_variable_mods(
        seq: str,
        match_group_to_index: Dict[str, int],
        var_mod_patterns: str,
        max_mods: int,
        id_to_name_map: Dict[int, str],
    ):

        mod_tups = [
            (match_group_to_index[match.group()], match.start())
            for match in re.finditer(var_mod_patterns, seq)
        ]

        # https://github.com/MannLabs/alphabase/issues/169
        # The modification site in alphabase (mod_sites) is
        # 1<=site<=n, 0 is for N-term, and -1 for C-term.
        get_site = lambda site: -1 if site == len(seq) - 1 else site

        num_mod_sites = len(mod_tups)
        comb_mod_ids, comb_mods, comb_mod_sites = [], [], []

        for r in range(1, max_mods + 1):
            for comb in combinations(range(num_mod_sites), r):
                comb_mod_ids.append(
                    MOD_SEPARATOR.join(map(str, (mod_tups[i][0] for i in comb)))
                )
                comb_mods.append(
                    MOD_SEPARATOR.join(id_to_name_map[mod_tups[i][0]] for i in comb)
                )
                comb_mod_sites.append(
                    MOD_SEPARATOR.join(
                        map(str, (get_site(mod_tups[i][1]) for i in comb))
                    )
                )

        return comb_mod_ids, comb_mods, comb_mod_sites

    def generate_static_modifications(self, peptide_df, use_mp=False):

        mod_param_set = self.mod_param_set
        static_mods = [p for p in mod_param_set if p.fixed]

        schema = {"mod_ids": pl.String, "mods": pl.String, "mod_sites": pl.String}

        if static_mods is None or len(static_mods) < 1:
            return peptide_df.with_columns(
                pl.lit(None).cast(v).alias(k) for k, v in schema.items()
            )

        fixed_mod_patterns = "|".join([m.get_re_pattern() for m in static_mods])
        tmp_df = peptide_df.select(pl.col("peptide_index", "peptide")).filter(
            pl.col("peptide").str.contains(fixed_mod_patterns)
        )

        id_to_name_map = {mod.id: mod.mod_name for mod in static_mods}

        _gen_static_mods = partial(
            self.generate_static_mods,
            static_mods=static_mods,
            id_to_name_map=id_to_name_map,
        )

        if use_mp:
            # Use 'spawn' context to avoid deadlocks with multi-threaded environments
            from delpi.utils.mp import get_multiprocessing_context

            with get_multiprocessing_context().Pool(
                processes=mp.cpu_count() // 2
            ) as pool:
                fixed_mod_list = pool.map(_gen_static_mods, tmp_df["peptide"])
        else:
            fixed_mod_list = [_gen_static_mods(seq) for seq in tmp_df["peptide"]]

        static_df = pl.from_records(
            fixed_mod_list, orient="row", schema=schema
        ).with_columns(tmp_df["peptide_index"])

        mod_df = peptide_df.select("peptide_index").join(
            static_df, on="peptide_index", how="left"
        )

        return mod_df

    def generate_variable_modifications(
        self,
        peptide_df: pl.DataFrame,
        use_mp: bool = False,
    ):
        mod_param_set = self.mod_param_set
        max_mods = self.max_mods
        var_mods = [p for p in mod_param_set if not p.fixed]

        if len(var_mods) < 1 or max_mods < 1:
            return None

        schema = {
            "var_mod_ids": pl.String,
            "var_mods": pl.String,
            "var_mod_sites": pl.String,
        }

        if var_mods is None or len(var_mods) < 1:
            return pl.DataFrame(schema=schema)

        var_mod_patterns = "|".join([m.get_re_pattern() for m in var_mods])
        match_group_to_index = {
            match_grp: mod.id for mod in var_mods for match_grp in mod.match_groups()
        }
        id_to_name_map = {mod.id: mod.mod_name for mod in var_mods}

        assert len(match_group_to_index) == len(
            set(match_group_to_index)
        ), "Multiple modifications cannot apply to the same amino acid"

        _generate_var_mods = partial(
            self.generate_variable_mods,
            max_mods=max_mods,
            match_group_to_index=match_group_to_index,
            var_mod_patterns=var_mod_patterns,
            id_to_name_map=id_to_name_map,
        )

        ######## find var modification sites ########
        m_peptide_df = peptide_df.filter(
            pl.col("peptide").str.contains(var_mod_patterns)
        ).select(pl.col("peptide_index", "peptide"))

        if use_mp:
            # Use 'spawn' context to avoid deadlocks with multi-threaded environments
            from delpi.utils.mp import get_multiprocessing_context

            with get_multiprocessing_context().Pool(
                processes=mp.cpu_count() // 2
            ) as pool:
                mod_peptide_list = pool.map(_generate_var_mods, m_peptide_df["peptide"])
        else:
            mod_peptide_list = [
                _generate_var_mods(seq) for seq in m_peptide_df["peptide"]
            ]

        var_mod_df = (
            pl.from_pandas(pd.DataFrame(mod_peptide_list, columns=list(schema)))
            .with_columns(m_peptide_df["peptide_index"])
            .explode(list(schema))
        )

        return var_mod_df

    def apply(
        self,
        peptide_df: pl.DataFrame,
        use_multiprocessing: bool = False,
    ):

        #### apply fixed modifications
        unmodified_df = self.generate_static_modifications(
            peptide_df, use_mp=use_multiprocessing
        )

        #### apply variable modifications
        var_mod_df = self.generate_variable_modifications(
            peptide_df, use_mp=use_multiprocessing
        )

        #### apply variable modifications
        if var_mod_df is not None:
            # merge with static modifications
            cols = []
            for mod_col in ["mod_ids", "mods", "mod_sites"]:
                cols.append(
                    pl.when(
                        pl.col(mod_col).is_null() & pl.col(f"var_{mod_col}").is_null()
                    )
                    .then(None)
                    .when(
                        pl.col(mod_col).is_not_null()
                        & pl.col(f"var_{mod_col}").is_not_null()
                    )
                    .then(pl.col(mod_col) + MOD_SEPARATOR + pl.col(f"var_{mod_col}"))
                    .when(pl.col(mod_col).is_not_null())
                    .then(pl.col(mod_col))
                    .otherwise(pl.col(f"var_{mod_col}"))
                    .alias(mod_col)
                )

            modified_df = (
                var_mod_df.join(unmodified_df, on="peptide_index", how="left")
                .select(pl.col("peptide_index"), cols[0], cols[1], cols[2])
                .sort(pl.col("peptide_index", "mods", "mod_sites"))
            )
            mod_df = pl.concat((unmodified_df, modified_df), how="vertical")
        else:
            mod_df = unmodified_df

        # mod_df = mod_df.with_row_index("peptidoform_index")
        return mod_df
