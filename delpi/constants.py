LOG_MSG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
LOG_DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

ELECTRON_MASS = 1.0 / 1822.8885020477
PROTON_MASS = 1.0072764667710
NEUTRON_MASS = 1.00866491566

C13C12_MASS_DIFF = 1.00335483

BOLTZMANN = 1.380657e-23

RESCALING_CONSTANT = 0.9995
RESCALING_CONSTANT_HIGH_PRECISION = 274.335215

# DDA precursor filtering around the isolation window
# Allow M+2 isotopes below the window and small transmission margins.
# I_min - ISOLATION_LOWER_TOL/charge < precursor_mz < I_max + ISOLATION_UPPER_TOL/charge
# DIA uses the isolation window without tolerance.
ISOLATION_LOWER_TOL = 3.5
ISOLATION_UPPER_TOL = 0.25

MAX_FRAGMENTS = 16
DEFAULT_Q_VALUE_CUTOFF = 0.01

QUANT_FRAGMENTS = 12
RT_WINDOW_RADIUS = 4
RT_WINDOW_LEN = 2 * RT_WINDOW_RADIUS + 1

DIA_MATCHED_PEAKS_CUTOFF = 9
DDA_MATCHED_PEAKS_CUTOFF_LB = 3
DDA_MATCHED_PEAKS_CUTOFF_UB = 6

PMSM_EMBEDDING_DIM = 192
# Number of PmSMs for training the TDA classifier
TDA_MAX_TRAIN_SIZE = 8 * 1024 * 1024

DEFAULT_REPORT_FORMAT = "parquet"  # tsv or parquet

# Default number of top-scoring observations per precursor (across runs) used
# to fine-tune the RT/MS2 predictors during the transfer-learning two-pass
# search.
DEFAULT_TL_TOP_K = 3

# Minimum stored fragments required for candidate searching.
MIN_SEARCH_FRAGMENTS = 5
DEFAULT_PRECURSOR_ISOTOPES = 4  # M-1, M, M+1, M+2
DEFAULT_FRAGMENT_ISOTOPES = 3  # M-1, M, M+1

DEFAULT_NCE = 30.0
