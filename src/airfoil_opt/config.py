"""
Single source of truth for all configuration parameters.
"""
from pathlib import Path

# --- DIRECTORY AND PATH SETTINGS ---

# XFOIL executable path
XFOIL_EXECUTABLE = "xfoil"

# Main output directories
IMAGES_DIR = Path("images")
DATA_DIR = Path("data")
OPTIMAL_DIR = Path("optimal")

# Working directories for XFOIL runs
DAT_DIR = Path("airfoil_dat_files")
POLAR_DIR = Path("polar_files")
INPUTS_DIR = Path("input_files")

# Directories for seed airfoil data
SEED_AIRFOILS_DIR = Path("seed_airfoils")
SEED_DAT_DIR = SEED_AIRFOILS_DIR / "dat_files"

# Pickle file paths
LHS_RESULTS_PATH = DATA_DIR / "lhs_results.pkl"
OPT_HISTORY_PATH = DATA_DIR / "optimization_history.pkl"
CTRL_HISTORY_PATH = DATA_DIR / "control_deltas_history.pkl"
SURROGATE_MODELS_DIR = DATA_DIR / "surrogate_models"

# --- XFOIL ANALYSIS PARAMETERS ---
ALPHA_START = -4.0
ALPHA_END = 18.0
ALPHA_STEP = 1.0
MACH_NUMBER = 0.20
REYNOLDS_NUMBER = 5e6
PANEL_POINTS = 120

# --- LHS (INITIAL SAMPLING) PARAMETERS ---
LHS_SAMPLES = 15
LHS_Y_BOUND = 0.08
LHS_X_BOUND = 0.05

# --- FFD (GEOMETRY DEFORMATION) PARAMETERS ---
FFD_NX = 10
FFD_NY = 6
FFD_PAD_X = 0.05
FFD_PAD_Y = 0.08
ENABLE_X_DOFS = True

# --- OPTIMIZER (SLSQP) PARAMETERS ---
OPT_Y_BOUND = 0.05
OPT_X_BOUND = 0.03
MAX_SLSQP_ITERS = 350
# increase to make optimization more creative
EXPLORATION_FACTOR = 0.9
FTOL = 1e-7
EPSILON = 1.5e-3

# --- ACTIVE LEARNING LOOP PARAMETERS ---
ACTIVE_MAX_ITERS = 15
REUSE_LHS_RESULTS = False
# increase to loosen optimization convergence criteria
ACTIVE_J_TOL = 1e-4
ACTIVE_UNCERTAINTY_TOL = 0.09

# --- ADVANCED GP HYPERPARAMETER OPTIMIZATION ---
USE_ADVANCED_CONVERGENCE = True
# increase to loosen regret tolerance
EI_REGRET_TOLERANCE = 7e-4
CONFIDENCE_LEVEL = 0.95
PAC_PROBABILITY_DELTA = 0.05

# --- OBJECTIVE FUNCTION WEIGHTS ---
WEIGHTS = {
    'w_alpha': 0.6,
    'w_cl': 0.8,
    'w_cd': 0.5,
    'w_cd0': 1.0,
    'w_cm': 2.0,
    'w_cm0': 0.5,
    'alpha_target': 15.0,
    'w_transition': 1.0,
    'cd0_target': 0.01,
    'cm_target': -0.02,
    'cm0_target': -0.09,
    'cm_tolerance': 0.0001,
    'w_t_upper': 40.0,
    'w_t_lower': 40.0,
    'w_overlap': 1e4,
    'xtr_min': 0.4,
    'w_recovery': 15.0,
    'w_detach': 20.0,
    'xtr_detach_min': 0.5,
}
