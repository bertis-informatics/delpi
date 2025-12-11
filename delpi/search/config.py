from typing import Union, List
from pathlib import Path

from delpi.chem.modification_param import ModificationParam
from delpi.lcms.reader.base import MassSpecFileReader
from delpi.search.result_manager import ResultManager
from delpi.utils.yaml_file import load_yaml, save_yaml

SUPPORTED_FILE_TYPES = (".raw", ".mzml", ".mzml.gz", "h5")


class SearchConfig:

    def __init__(self, yaml_path: Union[Path, str]):
        self.yaml_path = Path(yaml_path)
        self.config = load_yaml(self.yaml_path)
        self.bmsio_server_addr = "localhost:5001"
        self.input_files = self._get_input_files()

    def _get_input_files(self):

        input_files = []
        if "input_dir" in self.config:
            input_dir = Path(self.config["input_dir"])
            for ext in SUPPORTED_FILE_TYPES:
                input_files.extend(input_dir.glob(f"*{ext}", case_sensitive=False))
            if len(input_files) < 1:
                raise ValueError(f"Cannot find any input files from {input_dir}")

        elif "input_files" in self.config:
            for input_file in self.config["input_files"]:
                if input_file.lower().endswith(SUPPORTED_FILE_TYPES):
                    input_files.append(Path(input_file))
                else:
                    raise ValueError(f"Unsupported file type for {input_file}")
        else:
            raise ValueError("Missing 'input_files' or 'input_dir' in configuration")

        return input_files

    def set_bmsio_server(self, server_addr: str):
        self.bmsio_server_addr = server_addr

    def get_param(self, key: str):
        return self.config.get(key)

    def __getitem__(self, key):
        return self.config[key]

    def save(self, yaml_path: Union[Path, str]):
        save_yaml(yaml_path, self.config)

    @property
    def db_dir(self):
        return Path(self.config["database_directory"])

    @property
    def output_dir(self):
        return Path(self.config["output_directory"])

    @property
    def refined_db_dir(self):
        return self.output_dir / "refined_speclib"

    @property
    def log_file_path(self):
        return self.output_dir / "delpi.log"

    @property
    def fasta_file(self):
        return Path(self.config["fasta_file"])

    @property
    def run_names(self) -> List[str]:
        return [
            MassSpecFileReader.extract_run_name(input_file)
            for input_file in self.input_files
        ]

    def get_result_manager(self, run_name_or_index: Union[int, str]) -> ResultManager:

        if isinstance(run_name_or_index, int):
            input_file = self.input_files[run_name_or_index]
            run_name = MassSpecFileReader.extract_run_name(input_file)
        elif isinstance(run_name_or_index, str):
            run_name = run_name_or_index
        else:
            raise ValueError("run_name_or_index must be int or str")

        return ResultManager(run_name=run_name, output_dir=self.output_dir)

    def check_params(self) -> None:
        """
        Validate essential configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if "input_files" not in self.config and "input_dir" not in self.config:
            raise ValueError(
                "Missing required parameter 'input_files' or 'input_dir' in configuration"
            )

        # Check required parameters
        required_fields = ["output_directory", "database_directory"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(
                    f"Missing required parameter '{field}' in configuration"
                )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate q-value cutoff
        q_value_cutoff = self.get_param("q_value_cutoff") or 0.01
        if not 0 <= q_value_cutoff <= 1:
            raise ValueError(
                f"q_value_cutoff must be between 0 and 1, got {q_value_cutoff}"
            )

        # Validate decoy method
        decoy_method = self.get_param("decoy") or "pseudo_reverse"
        supported_decoys = ["pseudo_reverse", "diann"]
        if decoy_method not in supported_decoys:
            raise ValueError(
                f"Unsupported decoy method '{decoy_method}'. "
                f"Supported methods: {supported_decoys}"
            )

        if len(self.input_files) < 1:
            raise ValueError("No input files specified.")

        # Validate input files exist
        for input_file in self.input_files:
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

        # Validate fasta file exists
        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_file}")

    def get_db_params(self) -> dict:
        """
        Extract database-related parameters from configuration.

        Returns:
            dict: Dictionary containing only database-related parameters
        """
        db_params = {}

        # Database-related parameter keys
        db_param_keys = [
            "fasta_file",
            "digest",
            "modification",
            "precursor",
            "fragment",
        ]

        for key in db_param_keys:
            if key in self.config:
                db_params[key] = self.config[key]

        return db_params

    @property
    def is_phospho_search(self) -> bool:
        mod_params = self.config.get("modification", {}).get("mod_param_set", [])
        for mod in mod_params:
            if mod.get("mod_name", "").lower() == "phospho":
                return True
        return False

    def _compare_mod_params(self, mod_params1: list, mod_params2: list) -> bool:
        """
        Compare modification parameter lists.

        Args:
            mod_params1: First modification parameter list
            mod_params2: Second modification parameter list

        Returns:
            bool: True if parameters are identical, False otherwise
        """
        if len(mod_params1) != len(mod_params2):
            return False

        encoded1 = sorted([ModificationParam(**mods).encode() for mods in mod_params1])
        encoded2 = sorted([ModificationParam(**mods).encode() for mods in mod_params2])

        return encoded1 == encoded2

    def _find_diff_keys(self, d1: dict, d2: dict, prefix: str = "") -> List[str]:
        """
        Find keys with different values between two dictionaries.

        Args:
            d1: First dictionary
            d2: Second dictionary
            prefix: Prefix for nested keys

        Returns:
            List[str]: List of keys with different values
        """
        diffs = []
        all_keys = set(d1.keys()) & set(d2.keys())

        for key in all_keys:
            v1 = d1.get(key)
            v2 = d2.get(key)
            path = f"{prefix}.{key}" if prefix else key

            if key == "mod_param_set":
                if not self._compare_mod_params(v1, v2):
                    diffs.append(path)
            elif isinstance(v1, dict) and isinstance(v2, dict):
                diffs.extend(self._find_diff_keys(v1, v2, path))
            elif isinstance(v1, list) and isinstance(v2, list):
                min_len = min(len(v1), len(v2))
                for i in range(min_len):
                    if isinstance(v1[i], dict) and isinstance(v2[i], dict):
                        diffs.extend(self._find_diff_keys(v1[i], v2[i], f"{path}[{i}]"))
                    elif v1[i] != v2[i]:
                        diffs.append(f"{path}[{i}]")
                if len(v1) != len(v2):
                    diffs.append(path + f" (length mismatch)")
            elif v1 != v2:
                diffs.append(path)

        return diffs

    def compare_with_saved_params(self, db_dir: Path) -> List[str]:
        """
        Compare current database parameters with saved parameters.

        Args:
            db_dir: Database directory path

        Returns:
            List[str]: List of parameter keys that differ between current and saved params.
                      Empty list if parameters are identical.
        """
        param_file = db_dir / "param.yaml"

        saved_params = load_yaml(param_file)
        current_params = self.get_db_params()

        return self._find_diff_keys(saved_params, current_params)

    def check_database_exists(self) -> bool:
        """
        Check if database exists and parameters match current configuration.

        Returns:
            bool: True if database exists and parameters match

        Raises:
            ValueError: If database exists but parameters don't match
        """
        if not self.db_dir.exists():
            return False

        param_file = self.db_dir / "param.yaml"
        if not param_file.exists():
            return False

        diff_keys = self.compare_with_saved_params(self.db_dir)

        if len(diff_keys) > 0:
            raise ValueError(
                f"Database files already exist, but parameters used for DB generation are different with current params: {diff_keys}"
            )
        return True
