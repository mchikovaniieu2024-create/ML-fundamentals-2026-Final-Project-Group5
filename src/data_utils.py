import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
    FunctionTransformer
)

from sklearn.impute import SimpleImputer

from config import (
    DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    RANDOM_STATE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    EDUCATION_FEATURES,
    EXPERIENCE_FEATURES,
    SKILL_NUMERIC_COL,
    SKILL_TEXT_COL,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    EDLEVEL_ORDER,
)

from feature_engineering import (
    BinarySkillTransformer,
    SkillCountTransformer,
    GroupedSkillTransformer,
    EmbeddingTransformer,
)

from scipy import sparse


# Convenience mapping — teammates import this dict to get feature lists
FEATURE_GROUPS = {
    "education":  EDUCATION_FEATURES,
    "experience": EXPERIENCE_FEATURES,
    "skills":     [SKILL_TEXT_COL],
}


# 1. LOAD
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {len(df):,} rows × {df.shape[1]} columns from '{path}'")
    return df


# 2. CLEAN
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw dataframe.
    1. Drop leakage and index columns.
    2. Handle the one column with missing values.
    3. Standardise string whitespace across all object columns.
    4. Validate that the target column is binary.

    Leakage note
    PreviousSalary and Employment are dropped because:
      PreviousSalary implies the person was previously employed, directly
        leaking information about the target.
      Employment encodes current employment type, a proxy for the target.
    Both would have been available at training time but not at prediction time
    for a new, unseen candidate.

    """
    df = df.copy()

    # 2a. Drop leakage and index columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[clean_data] Dropped columns: {cols_to_drop}")

    # 2b. Missing value handling
    if "HaveWorkedWith" in df.columns:
        n_missing = df["HaveWorkedWith"].isna().sum()
        df["HaveWorkedWith"] = df["HaveWorkedWith"].fillna("")
        print(f"[clean_data] HaveWorkedWith: filled {n_missing} nulls with empty string")

    # 2c. Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 2d. Validate target
    assert df[TARGET_COL].isin([0, 1]).all(), \
        f"[clean_data] Target '{TARGET_COL}' contains values other than 0 and 1!"
    print(f"[clean_data] Target distribution:\n{df[TARGET_COL].value_counts()}")

    print(f"[clean_data] Done. Shape after cleaning: {df.shape}")
    return df


# 3. SPLIT

def get_split(df: pd.DataFrame):
    """
    Split the cleaned dataframe into train / validation / test sets.

    Split strategy
    70 % train | 15 % validation | 15 % test
    Stratification is applied on the target column so that the
    class ratio is preserved in every split.

    Avoid leakage
    The split is performed on the full cleaned dataframe before any
    encoding or scaling. Encoders are fitted only on the training set
    and applied to validation/test in build_pipeline(). This is the
    correct way to avoid target/distribution leakage.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - TRAIN_SIZE),
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: val vs test
    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - relative_val_size),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(
        f"[get_split] Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | "
        f"Test: {len(X_test):,}"
    )
    print(
        f"[get_split] Train class balance: "
        f"{y_train.value_counts(normalize=True).to_dict()}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# 4. PREPROCESSING PIPELINES

def _education_preprocessor() -> ColumnTransformer:
    """
    Preprocessor for education-only features.

    EdLevel → OrdinalEncoder with the natural degree ordering from config.
    All other categoricals → OneHotEncoder (handle_unknown='ignore' so
    unseen categories in val/test don't crash the pipeline).
    """
    ordinal_cols = ["EdLevel"]
    onehot_cols  = [c for c in EDUCATION_FEATURES
                    if c in CATEGORICAL_COLS and c not in ordinal_cols]

    return ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[EDLEVEL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_cols,
            ),
        ],
        remainder="drop",
    )


def _experience_preprocessor() -> ColumnTransformer:
    """
    Preprocessor for experience-only features.

    YearsCode and YearsCodePro are already numeric integers, StandardScaler
    centres and scales them so regularised models are not biased by raw magnitude.
    """
    return ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                EXPERIENCE_FEATURES,    # ["YearsCode", "YearsCodePro"]
            ),
        ],
        remainder="drop",
    )


def _skills_preprocessor(kind: str = "embeddings"):
    """
    Preprocessor for skills-only features.

    Supports four representations:
      - binary:     sparse binary indicator per skill
      - counts:     total distinct skill count
      - grouped:    technology group counts and binary flags
      - embeddings: dense sentence embeddings via all-MiniLM-L6-v2
    """
    if kind == "binary":
        skill_tf = BinarySkillTransformer(min_freq=2)
    elif kind == "counts":
        skill_tf = SkillCountTransformer()
    elif kind == "grouped":
        skill_tf = GroupedSkillTransformer(mode="both")
    elif kind == "embeddings":
        skill_tf = EmbeddingTransformer()
    else:
        raise ValueError(f"Unknown skills kind: {kind}")

    def _select_skill(X):
        if isinstance(X, pd.DataFrame):
            return X["HaveWorkedWith"]
        return X

    def _to_dense(X):
        if sparse.issparse(X):
            return X.toarray()
        return X

    return Pipeline([
        ("select_skill",  FunctionTransformer(_select_skill, validate=False)),
        ("skill_features", skill_tf),
        ("to_dense",       FunctionTransformer(_to_dense, validate=False)),
    ])


def _combined_preprocessor() -> ColumnTransformer:
    """
    Preprocessor for the combined model.

    Combines all numeric and categorical columns from all feature groups,
    excluding HaveWorkedWith.
    """
    ordinal_cols = ["EdLevel"]
    numeric_cols = EXPERIENCE_FEATURES + ["ComputerSkills"]
    onehot_cols  = (
        [c for c in EDUCATION_FEATURES if c in CATEGORICAL_COLS and c != "EdLevel"]
    )

    return ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[EDLEVEL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            (
                "scaler",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_cols,
            ),
        ],
        remainder="drop",
    )


def _baseline_preprocessor() -> ColumnTransformer:
    """
    Preprocessor for the baseline model.
    Combines education categoricals and experience numerics.
    """
    ordinal_cols = ["EdLevel"]
    onehot_cols  = [c for c in EDUCATION_FEATURES
                    if c in CATEGORICAL_COLS and c not in ordinal_cols]
    numeric_cols = EXPERIENCE_FEATURES

    return ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[EDLEVEL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            (
                "scaler",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_cols,
            ),
        ],
        remainder="drop",
    )


# Map group names to their preprocessor factory functions
_PREPROCESSOR_MAP = {
    "education":  _education_preprocessor,
    "experience": _experience_preprocessor,
    "skills":     _skills_preprocessor,
    "combined":   _combined_preprocessor,
    "baseline":   _baseline_preprocessor,
}


def build_pipeline(model, feature_group: str, skills_kind: str = "embeddings") -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessor → model.

    The preprocessor is chosen based on feature_group, ensuring each
    experiment uses only the columns belonging to that group.
    Fitting the Pipeline on X_train automatically fits the preprocessor
    on training data only, no leakage into val/test.

    Parameters
    ----------
    model          : any sklearn-compatible estimator
    feature_group  : one of 'education', 'experience', 'skills',
                     'combined', 'baseline'
    skills_kind    : only used when feature_group == 'skills'.
                     One of 'binary', 'counts', 'grouped', 'embeddings'.

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = build_pipeline(LogisticRegression(), "education")
    >>> pipe.fit(X_train, y_train)
    >>> pipe.predict(X_val)
    """
    if feature_group == "skills":
        preprocessor = _skills_preprocessor(skills_kind)
    else:
        if feature_group not in _PREPROCESSOR_MAP:
            raise ValueError(
                f"Unknown feature_group '{feature_group}'. "
                f"Choose from: {list(_PREPROCESSOR_MAP.keys())}"
            )
        preprocessor = _PREPROCESSOR_MAP[feature_group]()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model",        model),
        ]
    )
    return pipeline


if __name__ == "__main__":
    print("Running data_utils.py sanity check")

    # Load
    df_raw = load_data()

    # Clean
    df_clean = clean_data(df_raw)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(df_clean)

    # Verify no overlap between splits
    train_idx = set(X_train.index)
    val_idx   = set(X_val.index)
    test_idx  = set(X_test.index)
    assert len(train_idx & val_idx)  == 0, "OVERLAP between train and val!"
    assert len(train_idx & test_idx) == 0, "OVERLAP between train and test!"
    assert len(val_idx  & test_idx)  == 0, "OVERLAP between val and test!"
    print("[check] No index overlap between splits")

    from sklearn.linear_model import LogisticRegression
    for group in ["education", "experience", "skills", "combined"]:
        pipe = build_pipeline(LogisticRegression(max_iter=200), group)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        print(f"[check] '{group}' pipeline fitted and predicted {len(preds)} samples")

    print("All checks passed. data_utils.py is ready.")