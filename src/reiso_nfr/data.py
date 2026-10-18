"""Dataset loading and the fixed five-fold protocol shared by every model.

Protocol (Section V-B of the paper): the fold of each requirement is stored in the
``test_fold`` column. When fold k is the test fold, fold (k + 1) mod 5 is the
validation fold and the remaining three folds are the training set. The folds are
the same for every model and every seed; the seed only changes model training.
"""
from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "data", "dataset_with_test_fold.csv")
N_FOLDS = 5
SEEDS = (42, 123, 2024)


def clean_text(text) -> str:
    """Collapse whitespace and strip, as in the original experiments."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the 1,793 requirements with columns id, text, L1, L2, test_fold."""
    df = pd.read_csv(path)
    required = {"id", "text", "L1", "L2", "test_fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    assert df["id"].is_unique, "requirement ids must be unique"
    assert set(df["test_fold"].unique()) == set(range(N_FOLDS)), "test_fold must be 0..4"
    return df.reset_index(drop=True)


def label_info(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Sorted L2 labels, sorted L1 labels, and the L2 -> L1 parent map."""
    parents = df.groupby("L2")["L1"].unique()
    bad = parents[parents.map(len) > 1]
    if len(bad):
        raise ValueError(f"L2 labels with more than one parent: {list(bad.index)}")
    l2_to_l1 = {l2: p[0] for l2, p in parents.items()}
    return sorted(df["L2"].unique()), sorted(df["L1"].unique()), l2_to_l1


def fold_split(df: pd.DataFrame, test_fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) for one test fold of the rotating design."""
    val_fold = (test_fold + 1) % N_FOLDS
    test = df[df["test_fold"] == test_fold]
    val = df[df["test_fold"] == val_fold]
    train = df[~df["test_fold"].isin([test_fold, val_fold])]
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        from transformers import set_seed

        set_seed(seed)
    except ImportError:  # the SVM baseline runs without torch
        pass


def prediction_frame(test: pd.DataFrame, pred_l2: List[str], l2_to_l1: Dict[str, str], seed: int) -> pd.DataFrame:
    """Out-of-fold prediction rows in the common format used by evaluate.py."""
    out = test[["id", "test_fold", "L1", "L2"]].rename(columns={"L1": "gold_L1", "L2": "gold_L2"}).copy()
    out["pred_L2"] = list(pred_l2)
    out["pred_L1"] = out["pred_L2"].map(l2_to_l1)
    out["seed"] = seed
    return out


def save_oof(parts: List[pd.DataFrame], df: pd.DataFrame, out_dir: str, seed: int) -> str:
    """Concatenate the five test folds in dataset order and save them."""
    oof = pd.concat(parts).set_index("id").loc[df["id"]].reset_index()
    assert len(oof) == len(df), "every requirement must be predicted exactly once"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{seed}.csv")
    oof.to_csv(path, index=False)
    return path
