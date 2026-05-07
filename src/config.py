import os
from pathlib import Path

# Paths
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the raw dataset
DATA_PATH = BASE_DIR / "data" / "raw" / "stackoverflow_full.csv"

# Directory where processed artefacts (e.g. split CSVs, encoders) are saved
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Directory for evaluation outputs (plots, tables)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Reproducibility
RANDOM_STATE = 42


# Data split ratios
# 70 % train / 15 % validation / 15 % test
# With ~73 k rows this gives ~11 k samples for val and test, goof reliable metric estimates.

TRAIN_SIZE = 0.70
VAL_SIZE   = 0.15
TEST_SIZE  = 0.15


# Target column
TARGET_COL = "Employed"   # binary: 1 = employed, 0 = not employed


# Columns to drop before modelling
# - Unnamed: 0  → leftover CSV index, carries no information
# - PreviousSalary → LEAKAGE RISK: knowing a previous salary implies the
#   person was previously employed, which directly leaks the target signal.
# - Employment → LEAKAGE RISK: direct proxy for current employment status.
#   It is an integer encoding of employment type and would trivially predict
#   the target. Must be excluded from all feature sets.

DROP_COLS = ["Unnamed: 0", "PreviousSalary", "Employment"]


# Feature groups
# Three separate groups are trained independently before being combined.
# Education features — qualifications and background variables
EDUCATION_FEATURES = [
    "EdLevel",          # highest education level (categorical)
    "MainBranch",       # whether the respondent is a professional developer
    "Accessibility",    # disability / accessibility status (background)
    "MentalHealth",     # mental health status (background)
    "Gender",           # gender (background)
    "Age",              # age group (background)
]

# Experience features — years of coding, professional coding
EXPERIENCE_FEATURES = [
    "YearsCode",        # total years coding (numeric)
    "YearsCodePro",     # years coding professionally (numeric)

]

# Skills features — technology stack (will be parsed / vectorised by Jade)
SKILLS_FEATURES = [
    "HaveWorkedWith",   # semicolon-separated list of technologies
    "ComputerSkills",   # count of distinct technologies (numeric)
]


# Categorical columns that need encoding in education and combined pipelines
CATEGORICAL_COLS = [
    "EdLevel",
    "MainBranch",
    "Accessibility",
    "MentalHealth",
    "Gender",
    "Age",
    "Country",
]

# Numeric columns that need scaling
NUMERIC_COLS = [
    "YearsCode",
    "YearsCodePro",
    "ComputerSkills",
]


# Education level ordering for ordinal encoding
EDLEVEL_ORDER = [
    "NoFormalEducation",
    "Primary",
    "Secondary",
    "SomeCollege",
    "Associate",
    "Undergraduate",
    "Master",
    "Professional",
    "PhD",
]
