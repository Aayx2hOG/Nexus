"""Create multiclass targets for the preprocessed UNSW-NB15 feature matrices.

This script reuses the official split and writes only targets and label metadata;
the feature matrices produced by ``preprocess_unsw_nb15.py`` are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CSV_Files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR_NAME = "Training and Testing Sets"
TRAIN_FILE_NAME = "UNSW_NB15_training-set.csv"
TEST_FILE_NAME = "UNSW_NB15_testing-set.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_labels(path: Path) -> tuple[pd.Series, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    frame = pd.read_csv(path, usecols=["attack_cat", "label"], low_memory=False)
    if frame[["attack_cat", "label"]].isna().any().any():
        raise ValueError(f"Missing target values in {path}")

    categories = frame["attack_cat"].astype(str).str.strip()
    binary = frame["label"].to_numpy(dtype=np.int64)
    expected_binary = (categories.str.casefold() != "normal").to_numpy(dtype=np.int64)
    if not np.array_equal(binary, expected_binary):
        raise ValueError(f"attack_cat and binary label disagree in {path}")
    return categories, binary


def prepare_targets(raw_dir: Path, output_dir: Path) -> dict[str, object]:
    split_dir = raw_dir / SPLIT_DIR_NAME
    train_categories, _ = read_labels(split_dir / TRAIN_FILE_NAME)
    test_categories, _ = read_labels(split_dir / TEST_FILE_NAME)

    train_classes = set(train_categories.unique())
    unseen_test_classes = sorted(set(test_categories.unique()) - train_classes)
    if unseen_test_classes:
        raise ValueError(f"Test set contains classes absent from training: {unseen_test_classes}")

    # Keep Normal first for readability, then use deterministic alphabetical order.
    class_names = sorted(train_classes, key=lambda value: (value.casefold() != "normal", value))
    class_to_id = {name: index for index, name in enumerate(class_names)}
    y_train = train_categories.map(class_to_id).to_numpy(dtype=np.int64)
    y_test = test_categories.map(class_to_id).to_numpy(dtype=np.int64)

    metadata: dict[str, object] = {
        "target_column": "attack_cat",
        "class_names": class_names,
        "class_to_id": class_to_id,
        "train_rows": int(y_train.size),
        "test_rows": int(y_test.size),
        "target_counts": {
            "train": {
                name: int((train_categories == name).sum()) for name in class_names
            },
            "test": {name: int((test_categories == name).sum()) for name in class_names},
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "y_train_multiclass.npy", y_train)
    np.save(output_dir / "y_test_multiclass.npy", y_test)
    (output_dir / "multiclass_labels.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = prepare_targets(args.raw_dir.resolve(), args.output_dir.resolve())
    print(
        f"Created multiclass targets for {len(metadata['class_names'])} classes "
        f"in {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
