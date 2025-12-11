from typing import List, Union
import logging
import warnings
from pathlib import Path
import torch

from delpi.lcms.base_ion_type import BaseIonType
from delpi.database.peptide_database import PeptideDatabase
from delpi.search.config import SearchConfig
from delpi.utils.log_config import configure_logging
from delpi.utils.yaml_file import save_yaml
from delpi.constants import MAX_FRAGMENTS


logger = logging.getLogger(__name__)


def build_database(
    search_config: SearchConfig,
    device: torch.device,
):
    configure_logging(logfile_path=search_config.log_file_path, level=logging.INFO)

    logger.info(f"Building database with {search_config['fasta_file']}")

    db_dir = Path(search_config["database_directory"])
    if db_dir.exists():
        logger.debug(f"Database directory already exists")
    else:
        logger.debug(f"Create database directory: {db_dir}")
        db_dir.mkdir()

    db = PeptideDatabase().build(
        fasta_file=search_config["fasta_file"],
        enzyme=search_config["digest"]["enzyme"],
        min_len=search_config["digest"]["min_len"],
        max_len=search_config["digest"]["max_len"],
        max_missed_cleavages=search_config["digest"]["max_missed_cleavages"],
        n_term_methionine_excision=search_config["digest"][
            "n_term_methionine_excision"
        ],
        # decoy="pseudo_reverse",
        decoy="diann",
        mod_param_set=search_config["modification"]["mod_param_set"],
        max_mods=search_config["modification"]["max_mods"],
        min_precursor_charge=search_config["precursor"].get("min_charge", 2),
        max_precursor_charge=search_config["precursor"].get("max_charge", 4),
        min_precursor_mz=search_config["precursor"].get("min_mz", 300),
        max_precursor_mz=search_config["precursor"].get("max_mz", 1800),
        min_fragment_charge=search_config["fragment"].get("min_charge", 1),
        max_fragment_charge=search_config["fragment"].get("max_charge", 2),
        min_fragment_mz=search_config["fragment"].get("min_mz", 200),
        max_fragment_mz=search_config["fragment"].get("max_mz", 1800),
        prefix_ion_type=BaseIonType.B,
        suffix_ion_type=BaseIonType.Y,
        max_fragments=MAX_FRAGMENTS,
        device=device,
        use_multiprocessing=True,
    )

    db.save(save_dir=db_dir)
    logger.info(f"Complete building database, saved to: {db_dir}")


def build_database_in_subprocess(search_config: SearchConfig, device: torch.device):
    # Use 'spawn' instead of 'fork' to avoid deadlocks with Numba's threading
    # when the main process is already multi-threaded (e.g., from Numba JIT compilation)
    from delpi.utils.mp import get_multiprocessing_context

    p = get_multiprocessing_context().Process(
        target=build_database,
        args=(search_config, device),
    )
    p.start()
    logger.debug(f"Start a child process (PID: {p.pid})")
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Database build failed (exit code: {p.exitcode})")

    logger.debug(f"Terminate child process (PID: {p.pid})")
