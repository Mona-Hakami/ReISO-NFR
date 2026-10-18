"""Character TF-IDF + LinearSVM baseline.

The vectorizer is fitted on the three training folds only. The validation fold is
not needed because the model has no early stopping or tuned settings.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from ..data import N_FOLDS, fold_split, prediction_frame, seed_everything


@dataclass
class SVMConfig:
    char_ngram_range: tuple = (3, 5)
    min_df: int = 2
    C: float = 1.0
    class_weight: str = "balanced"
    max_iter: int = 10000


def run(df: pd.DataFrame, l2_to_l1, seed: int, cfg: SVMConfig = SVMConfig()) -> list:
    parts = []
    for k in range(N_FOLDS):
        seed_everything(seed)
        train, _, test = fold_split(df, k)
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=cfg.char_ngram_range, lowercase=True,
            sublinear_tf=True, min_df=cfg.min_df, norm="l2",
        )
        x_train = vec.fit_transform(train["text"])
        clf = LinearSVC(C=cfg.C, class_weight=cfg.class_weight, random_state=seed, max_iter=cfg.max_iter)
        clf.fit(x_train, train["L2"])
        parts.append(prediction_frame(test, clf.predict(vec.transform(test["text"])), l2_to_l1, seed))
    return parts
