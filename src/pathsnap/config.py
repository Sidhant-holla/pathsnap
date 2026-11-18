"""Project-wide paths and constants.

Every module imports paths from here, so nobody hardcodes a folder. The raw data
folder can be overridden with the PATHSNAP_DATA environment variable (useful on the
API laptop, where the evaluator trips may live somewhere else).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Raw data: <DATA_ROOT>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "data" / "raw"))

PROCESSED_DIR = REPO_ROOT / "data" / "processed"   # feature tables (parquet), not committed
RESULTS_DIR = REPO_ROOT / "experiments" / "results"  # metrics JSON/CSV, committed
MODELS_DIR = REPO_ROOT / "models"                    # saved models, committed
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
MAPS_DIR = REPO_ROOT / "reports" / "maps"

TRAIN = "train"
STUDENT_TEST = "student_test"
EVALUATOR_TEST = "evaluator_test"   # final demo only, never used during development
PARTITIONS = (TRAIN, STUDENT_TEST, EVALUATOR_TEST)
DEV_PARTITIONS = (TRAIN, STUDENT_TEST)

SEED = 42
