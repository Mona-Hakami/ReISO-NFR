"""Evaluation metrics (Section V-B, Eqs. 14-15)."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_metrics(gold_l2: Iterable[str], pred_l2: Iterable[str], l2_to_l1: Dict[str, str]) -> Dict[str, float]:
    """Level-2, Level-1 and hierarchical metrics plus the error decomposition.

    hF1: each label is augmented with its parent; with one label and one parent per
    requirement, hP = hR = hF1 = mean of Level-2 and Level-1 accuracy.
    Errors are split into correct Level-2, wrong sibling (correct parent) and wrong parent.
    """
    t2, p2 = np.asarray(list(gold_l2)), np.asarray(list(pred_l2))
    t1 = np.array([l2_to_l1[x] for x in t2])
    p1 = np.array([l2_to_l1[x] for x in p2])
    c2, c1 = t2 == p2, t1 == p1
    return {
        "l2_macro_f1": f1_score(t2, p2, average="macro", zero_division=0),
        "l2_weighted_f1": f1_score(t2, p2, average="weighted", zero_division=0),
        "l2_accuracy": accuracy_score(t2, p2),
        "l2_balanced_accuracy": balanced_accuracy_score(t2, p2),
        "l1_macro_f1": f1_score(t1, p1, average="macro", zero_division=0),
        "l1_accuracy": accuracy_score(t1, p1),
        "hF1": (c1.sum() + c2.sum()) / (2 * len(t2)),
        "correct_l2_pct": 100 * c2.mean(),
        "wrong_sibling_pct": 100 * (~c2 & c1).mean(),
        "wrong_parent_pct": 100 * (~c1).mean(),
    }


def fold_mean_metrics(oof, l2_to_l1: Dict[str, str]) -> Dict[str, float]:
    """Metrics computed on each test fold, then averaged over the five folds."""
    rows: List[Dict[str, float]] = [
        compute_metrics(g["gold_L2"], g["pred_L2"], l2_to_l1) for _, g in oof.groupby("test_fold")
    ]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
