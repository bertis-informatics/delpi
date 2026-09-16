"""Search-specific registry of modifications (UniMod + custom).

A :class:`ModificationRegistry` wraps the global, process-wide, UniMod-derived
:class:`~delpi.chem.modification.Modification` catalogue and adds *custom*
modifications defined by elemental composition in a search configuration,
without ever mutating the global catalogue (``Modification.register`` is
never called for custom entries).

Instances only hold plain data (name/composition/accession strings & simple
objects), so they are picklable and can be created once in the main process
and handed to run-level child processes (fork/forkserver/spawn) unchanged.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from delpi.chem.composition import Composition
from delpi.chem.modification import Modification

# Custom modifications get internal accession numbers from this range so
# they can never collide with a real UniMod record id (currently < 3000).
# These numbers are only used internally (e.g. for bit-packing in
# ModificationParam.encode()) and are never surfaced as "UniMod:<id>".
CUSTOM_MOD_ID_START = 60_000
CUSTOM_MOD_ID_END = 65_535


class ModificationRegistry:
    """Resolves modification names to :class:`Modification` objects for a
    single search, combining the global UniMod catalogue with custom,
    composition-defined modifications declared in the search configuration.
    """

    def __init__(self, custom_mod_defs: Optional[Sequence[Dict[str, Any]]] = None):

        self._custom_by_name: Dict[str, Modification] = {}
        self._custom_by_id: Dict[int, Modification] = {}

        custom_mod_defs = list(custom_mod_defs or [])

        # Assign accession numbers in name-sorted (not declaration) order so
        # that two registries built from the same *set* of custom mod names
        # (e.g. listed in a different order in the config) are equivalent.
        by_name: Dict[str, Dict[str, Any]] = {}
        for mod_def in custom_mod_defs:
            name = mod_def.get("mod_name")
            if not name:
                raise ValueError("Custom modification is missing 'mod_name'")

            key = name.lower()
            if key.startswith("unimod:"):
                raise ValueError(
                    f"Invalid custom modification name '{name}': names starting "
                    "with 'UniMod:' are reserved for the UniMod database"
                )
            if key in by_name:
                if by_name[key].get("composition") != mod_def.get("composition"):
                    raise ValueError(
                        f"Conflicting definitions for custom modification '{name}': "
                        f"'{by_name[key].get('composition')}' vs '{mod_def.get('composition')}'"
                    )
                raise ValueError(f"Duplicate custom modification name: '{name}'")
            by_name[key] = mod_def

        next_id = CUSTOM_MOD_ID_START

        for key in sorted(by_name):
            mod_def = by_name[key]
            name = mod_def["mod_name"]
            composition_str = mod_def.get("composition")
            if not composition_str:
                raise ValueError(
                    f"Custom modification '{name}' is missing 'composition'"
                )

            try:
                existing = Modification.get(name)
            except KeyError:
                existing = None
            if existing is not None:
                raise ValueError(
                    f"Custom modification name '{name}' conflicts with an "
                    "existing UniMod modification. Choose a different name."
                )

            try:
                composition = Composition.parse_from_plain_string(composition_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid composition '{composition_str}' for custom "
                    f"modification '{name}': {e}"
                ) from e

            if next_id > CUSTOM_MOD_ID_END:
                raise ValueError(
                    "Too many custom modifications defined "
                    f"(max {CUSTOM_MOD_ID_END - CUSTOM_MOD_ID_START + 1})"
                )

            mod = Modification(
                accession_num=next_id,
                composition=composition,
                name=name,
                description=f"Custom modification ({composition_str})",
                is_custom=True,
            )
            self._custom_by_name[key] = mod
            self._custom_by_id[next_id] = mod
            next_id += 1

    @classmethod
    def from_mod_param_set(
        cls, mod_param_set: Optional[Sequence[Dict[str, Any]]]
    ) -> "ModificationRegistry":
        """Build a registry from a raw ``mod_param_set`` (as found in a
        search config or a saved ``param.yaml``), picking out entries that
        define a custom (composition-based) modification."""
        custom_mod_defs = [p for p in (mod_param_set or []) if p.get("composition")]
        return cls(custom_mod_defs)

    @property
    def has_custom_modifications(self) -> bool:
        return len(self._custom_by_name) > 0

    def get(self, mod_name: str) -> Modification:
        """Resolve a modification name, checking custom modifications first."""
        key = mod_name.lower()
        if key in self._custom_by_name:
            return self._custom_by_name[key]
        return Modification.get(mod_name)

    def get_by_id(self, accession_num: int) -> Modification:
        if accession_num in self._custom_by_id:
            return self._custom_by_id[accession_num]
        return Modification.get_by_unimod_id(accession_num)

    def is_custom_id(self, accession_num: int) -> bool:
        return accession_num in self._custom_by_id

    def get_display_name(self, accession_num: int) -> str:
        return self.get_by_id(accession_num).name

    def build_mod_mass_array(self) -> np.ndarray:
        """Build a dense accession_num -> mass lookup array covering both the
        global UniMod catalogue and this registry's custom modifications."""
        max_id = Modification.get_max_accession_num()
        if self._custom_by_id:
            max_id = max(max_id, max(self._custom_by_id))

        mod_mass_array = np.zeros(max_id + 1, dtype=np.float64)
        for mod in Modification.name_to_mod_map.values():
            mod_mass_array[mod.accession_num] = mod.mass
        for mod in self._custom_by_id.values():
            mod_mass_array[mod.accession_num] = mod.mass
        return mod_mass_array
