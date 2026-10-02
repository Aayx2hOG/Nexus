"""Prepare the official UNSW-NB15 split for tree and FT-Transformer models.

The fitted transformations use the training set only.  Run from the repository
root with::

    python preprocessing/preprocess_unsw_nb15.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data" / "raw" / "CSV_Files"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed"
SPLIT_DIR_NAME = "Training and Testing Sets"
TRAIN_FILE_NAME = "UNSW_NB15_training-set.csv"
TEST_FILE_NAME = "UNSW_NB15_testing-set.csv"
FEATURE_CATALOG_NAME = "NUSW-NB15_features.csv"

TARGET_COLUMN = "label"
# attack_cat determines whether label is zero or one, so using it as an input
# would leak the target. id is only a row identifier in the prepared split.
EXCLUDED_FEATURE_COLUMNS = ("id", "attack_cat")
UNKNOWN_CATEGORY_CODE = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing the UNSW-NB15 CSV files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated arrays (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        # The distributed feature catalog contains Windows-1252 punctuation.
        frame = pd.read_csv(path, encoding="cp1252", low_memory=False)
    frame.columns = frame.columns.str.strip()
    return frame


def load_categorical_columns(feature_catalog_path: Path, columns: list[str]) -> list[str]:
    """Use the supplied feature catalog to identify nominal predictors."""
    catalog = _read_csv(feature_catalog_path)
    required = {"Name", "Type"}
    if not required.issubset(catalog.columns):
        raise ValueError(
            f"{feature_catalog_path} must contain columns {sorted(required)}; "
            f"found {catalog.columns.tolist()}"
        )

    nominal_names = set(
        catalog.loc[
            catalog["Type"].astype(str).str.strip().str.casefold() == "nominal", "Name"
        ]
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    return [
        column
        for column in columns
        if column.casefold() in nominal_names
        and column not in EXCLUDED_FEATURE_COLUMNS
        and column != TARGET_COLUMN
    ]


def validate_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError("Training and testing CSV columns differ or are in a different order")
    required = {TARGET_COLUMN, *EXCLUDED_FEATURE_COLUMNS}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if train.empty or test.empty:
        raise ValueError("Training and testing sets must both contain rows")
    if train[TARGET_COLUMN].isna().any() or test[TARGET_COLUMN].isna().any():
        raise ValueError("Target labels may not be missing")

    labels = set(pd.concat([train[TARGET_COLUMN], test[TARGET_COLUMN]]).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, found: {sorted(labels)}")


def clean_categorical(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # A reserved token prevents actual missing values from becoming the string "nan".
    return frame[columns].fillna("__MISSING__").astype(str)


def fit_ft_category_maps(
    train_categories: pd.DataFrame,
) -> tuple[dict[str, dict[str, int]], list[int]]:
    mappings: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for column in train_categories.columns:
        values = sorted(train_categories[column].unique().tolist())
        mappings[column] = {value: code for code, value in enumerate(values, start=1)}
        cardinalities.append(len(values) + 1)  # code 0 is unknown/test-only
    return mappings, cardinalities


def apply_ft_category_maps(
    categories: pd.DataFrame, mappings: dict[str, dict[str, int]]
) -> np.ndarray:
    encoded = np.empty((len(categories), len(mappings)), dtype=np.int64)
    for index, (column, mapping) in enumerate(mappings.items()):
        encoded[:, index] = (
            categories[column].map(mapping).fillna(UNKNOWN_CATEGORY_CODE).astype(np.int64)
        )
    return encoded


def build_metadata(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    tree_feature_names: list[str],
    ft_cardinalities: list[int],
    category_maps: dict[str, dict[str, int]],
) -> dict[str, Any]:
    unseen_test_categories = {
        column: sorted(set(test[column].dropna().astype(str)) - set(category_maps[column]))
        for column in categorical_columns
    }
    return {
        "source": {
            "feature_catalog": FEATURE_CATALOG_NAME,
            "train": TRAIN_FILE_NAME,
            "test": TEST_FILE_NAME,
        },
        "train_rows": len(train),
        "test_rows": len(test),
        "target_column": TARGET_COLUMN,
        "excluded_feature_columns": list(EXCLUDED_FEATURE_COLUMNS),
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "tree_feature_names": tree_feature_names,
        "ft_categorical_cardinalities": ft_cardinalities,
        "ft_unknown_category_code": UNKNOWN_CATEGORY_CODE,
        "ft_category_maps": category_maps,
        "unseen_test_categories": unseen_test_categories,
        "target_counts": {
            "train": {
                str(key): int(value)
                for key, value in train[TARGET_COLUMN].value_counts().sort_index().items()
            },
            "test": {
                str(key): int(value)
                for key, value in test[TARGET_COLUMN].value_counts().sort_index().items()
            },
        },
    }


def preprocess(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    split_dir = raw_dir / SPLIT_DIR_NAME
    train = _read_csv(split_dir / TRAIN_FILE_NAME)
    test = _read_csv(split_dir / TEST_FILE_NAME)
    validate_splits(train, test)

    candidate_columns = [
        column
        for column in train.columns
        if column not in {*EXCLUDED_FEATURE_COLUMNS, TARGET_COLUMN}
    ]
    categorical_columns = load_categorical_columns(
        raw_dir / FEATURE_CATALOG_NAME, candidate_columns
    )
    numerical_columns = [
        column for column in candidate_columns if column not in categorical_columns
    ]
    if not categorical_columns:
        raise ValueError("No categorical predictors were identified from the feature catalog")

    train_cat = clean_categorical(train, categorical_columns)
    test_cat = clean_categorical(test, categorical_columns)

    numeric_imputer = SimpleImputer(strategy="median")
    train_num_unscaled = numeric_imputer.fit_transform(train[numerical_columns]).astype(
        np.float32
    )
    test_num_unscaled = numeric_imputer.transform(test[numerical_columns]).astype(np.float32)

    if not np.isfinite(train_num_unscaled).all() or not np.isfinite(test_num_unscaled).all():
        raise ValueError("Numerical predictors contain infinite values")

    # RF and LightGBM share a sparse float32 matrix. One-hot encoding avoids
    # imposing a false ordinal relationship on protocol/service/state values.
    tree_encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float32
    )
    train_tree_cat = tree_encoder.fit_transform(train_cat)
    test_tree_cat = tree_encoder.transform(test_cat)
    x_train_tree = sparse.hstack(
        [sparse.csr_matrix(train_num_unscaled), train_tree_cat], format="csr"
    )
    x_test_tree = sparse.hstack(
        [sparse.csr_matrix(test_num_unscaled), test_tree_cat], format="csr"
    )
    tree_feature_names = numerical_columns + tree_encoder.get_feature_names_out(
        categorical_columns
    ).tolist()

    # FT-Transformer receives one integer token per categorical field and a
    # separately standardized numerical matrix. Unknown categories use code 0.
    category_maps, ft_cardinalities = fit_ft_category_maps(train_cat)
    x_train_ft_cat = apply_ft_category_maps(train_cat, category_maps)
    x_test_ft_cat = apply_ft_category_maps(test_cat, category_maps)
    numeric_scaler = StandardScaler()
    x_train_ft_num = numeric_scaler.fit_transform(train_num_unscaled).astype(np.float32)
    x_test_ft_num = numeric_scaler.transform(test_num_unscaled).astype(np.float32)

    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "X_train_tree.npz", x_train_tree, compressed=True)
    sparse.save_npz(output_dir / "X_test_tree.npz", x_test_tree, compressed=True)
    np.save(output_dir / "X_train_ft_cat.npy", x_train_ft_cat)
    np.save(output_dir / "X_train_ft_num.npy", x_train_ft_num)
    np.save(output_dir / "X_test_ft_cat.npy", x_test_ft_cat)
    np.save(output_dir / "X_test_ft_num.npy", x_test_ft_num)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    metadata = build_metadata(
        train,
        test,
        categorical_columns,
        numerical_columns,
        tree_feature_names,
        ft_cardinalities,
        category_maps,
    )
    with (output_dir / "preprocessing_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    joblib.dump(
        {
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "numeric_imputer": numeric_imputer,
            "tree_encoder": tree_encoder,
            "ft_category_maps": category_maps,
            "ft_numeric_scaler": numeric_scaler,
        },
        output_dir / "preprocessor.joblib",
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = preprocess(args.raw_dir.resolve(), args.output_dir.resolve())
    print(
        "Created processed UNSW-NB15 arrays in "
        f"{args.output_dir.resolve()} ({metadata['train_rows']} train rows, "
        f"{metadata['test_rows']} test rows)."
    )


if __name__ == "__main__":
    main()
