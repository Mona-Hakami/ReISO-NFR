"""Full ReISO-NFR run (Stage 1 + Stage 2) over the five folds for one seed, and the
ablation variants of Table 14."""
from __future__ import annotations

import gc
from dataclasses import replace
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

from . import reranker as rr
from .cards import load_cards
from .data import N_FOLDS, fold_split, prediction_frame, seed_everything
from .retrieval import RetrievalConfig, encode, fold_rankings, load_retriever

# Table 14: each variant changes one component of the full model.
VARIANTS = {
    "full": {},
    "pointwise": {"loss": "pointwise"},
    "no_margin": {"margin_lambda": 0.0},
    "no_hard_negatives": {"hard_negatives": 0},
    "iso_only": {"mode": "iso_only"},          # alpha = 1
    "knn_only": {"mode": "knn_only"},          # alpha = 0
    "iso_definitions_only": {"definitions_only": True},
}
HARD_NEGATIVE_SEED_OFFSET = 10000


def configs(variant: str) -> Tuple[RetrievalConfig, rr.RerankerConfig, bool]:
    change = dict(VARIANTS[variant])
    definitions_only = change.pop("definitions_only", False)
    ret = replace(RetrievalConfig(), **{k: v for k, v in change.items() if k in RetrievalConfig.__dataclass_fields__})
    rer = replace(rr.RerankerConfig(), **{k: v for k, v in change.items() if k in rr.RerankerConfig.__dataclass_fields__})
    return ret, rer, definitions_only


def run(df: pd.DataFrame, l2_labels: List[str], l2_to_l1: Dict[str, str], seed: int,
        variant: str = "full") -> Tuple[list, list]:
    """Return (ReISO-NFR predictions, Stage-1-only predictions) for the five test folds."""
    ret_cfg, rr_cfg, defs_only = configs(variant)
    retrieval_cards, reranker_cards = load_cards(l2_labels, definitions_only=defs_only)
    seed_everything(seed)  # once per run, as in the original experiments
    retriever = load_retriever(ret_cfg)
    label_emb = encode(retriever, [retrieval_cards[l] for l in l2_labels], ret_cfg.batch_size, is_query=False)
    tok = AutoTokenizer.from_pretrained(rr_cfg.model_name, use_fast=True)
    parts, stage1_parts = [], []
    for k in range(N_FOLDS):
        train, val, test = fold_split(df, k)
        ranked = fold_rankings(retriever, train, val, test, l2_labels, label_emb, ret_cfg)
        mk = lambda d, name, is_train, s=0: rr.CandidateDataset(d, ranked[name], reranker_cards, tok, rr_cfg,
                                                                l2_to_l1, train=is_train, seed=s)
        train_ds = mk(train, "train", True, seed + HARD_NEGATIVE_SEED_OFFSET + 1000 * k)
        val_ds, test_ds = mk(val, "val", False), mk(test, "test", False)
        weights = rr.class_weights(train["L2"].tolist(), l2_labels, rr_cfg.class_weight_clip)
        model = rr.train(rr.new_model(rr_cfg), train_ds, val_ds, weights, l2_to_l1, l2_labels, rr_cfg)
        pred = prediction_frame(test, rr.predict(model, test_ds, rr_cfg), l2_to_l1, seed)
        pred["candidates"] = ["|".join(c[: ret_cfg.candidate_k]) for c in ranked["test"]]
        parts.append(pred)
        stage1_parts.append(prediction_frame(test, [c[0] for c in ranked["test"]], l2_to_l1, seed))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return parts, stage1_parts
