from typing import List, Union
import logging
import queue
import warnings
from pathlib import Path
import torch

from delpi.lcms.base_ion_type import BaseIonType
from delpi.database.peptide_database import PeptideDatabase
from delpi.search.config import SearchConfig
from delpi.search.progress import CallbackProgressTracker, TqdmProgressTracker
from delpi.utils.log_config import configure_logging
from delpi.utils.yaml_file import save_yaml
from delpi.constants import MAX_FRAGMENTS

logger = logging.getLogger(__name__)


def build_database(
    search_config: SearchConfig,
    device: torch.device,
    batch_size: int = 512,
    progress_queue=None,
):
    use_tqdm = progress_queue is None
    configure_logging(
        logfile_path=search_config.log_file_path, level=logging.INFO, use_tqdm=use_tqdm
    )

    logger.info(f"Building database with {search_config['fasta_file']}")

    db_dir = Path(search_config["database_directory"])
    if db_dir.exists():
        logger.debug(f"Database directory already exists")
    else:
        logger.debug(f"Create database directory: {db_dir}")
        db_dir.mkdir()

    if progress_queue is not None:
        progress = CallbackProgressTracker(
            total=100,
            description="Building database",
            callback=lambda snap: progress_queue.put(snap),
        )
    else:
        progress = TqdmProgressTracker(total=100, description="Building database")

    try:
        db = PeptideDatabase().build(
            fasta_file=search_config["fasta_file"],
            enzyme=search_config["digest"]["enzyme"],
            min_len=search_config["digest"]["min_len"],
            max_len=search_config["digest"]["max_len"],
            max_missed_cleavages=search_config["digest"]["max_missed_cleavages"],
            n_term_methionine_excision=search_config["digest"][
                "n_term_methionine_excision"
            ],
            decoy=search_config.config.get("decoy_method", "mutation"),
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
            precursor_chunk_size=65_536,
            batch_size=batch_size,
            progress=progress,
        )

        db.save(save_dir=db_dir)
        logger.info(f"Complete building database, saved to: {db_dir}")
    finally:
        progress.close()
        if progress_queue is not None:
            progress_queue.put(None)  # sentinel — tells parent we're done


def build_database_in_subprocess(
    search_config: SearchConfig,
    device: torch.device,
    batch_size: int = 512,
    progress: CallbackProgressTracker = None,
):
    # Use 'spawn' instead of 'fork' to avoid deadlocks with Numba's threading
    # when the main process is already multi-threaded (e.g., from Numba JIT compilation)
    from delpi.utils.mp import get_multiprocessing_context

    mp_ctx = get_multiprocessing_context()
    progress_queue = mp_ctx.Queue() if progress is not None else None

    p = mp_ctx.Process(
        target=build_database,
        args=(search_config, device, batch_size),
        kwargs={"progress_queue": progress_queue} if progress_queue else {},
    )
    p.start()
    logger.debug(f"Start a child process (PID: {p.pid})")

    if progress_queue is not None:
        # Drain snapshots from the child and forward to the caller's tracker
        while True:
            try:
                snapshot = progress_queue.get(timeout=1.0)
            except queue.Empty:
                if not p.is_alive():
                    break
                continue
            if snapshot is None:
                break
            progress.forward_snapshot(snapshot)

    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Database build failed (exit code: {p.exitcode})")

    logger.debug(f"Terminate child process (PID: {p.pid})")
