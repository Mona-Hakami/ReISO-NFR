"""ISO-HGCLR: RoBERTa with a Level-2 head, a Level-1 auxiliary head and a
hierarchy-guided supervised contrastive loss (Eq. 13).

Positives share the same Level-2 label; sibling negatives (same parent, different
Level-2 label) get a larger weight in the denominator than cross-parent negatives.
Final predictions use top-3 parent routing: only Level-2 labels whose parent is among
the three highest-scoring parents of the auxiliary head are allowed.
"""
from __future__ import annotations

import gc
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer

from ..common_torch import FixLayerNormKeys, FocalLoss, TextDataset, balanced_class_weights, training_args
from ..data import N_FOLDS, fold_split, prediction_frame, seed_everything


@dataclass
class HGCLRConfig:
    model_name: str = "roberta-base"
    max_len: int = 128
    batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.08
    max_epochs: int = 30
    patience: int = 3
    encoder_dropout: float = 0.15
    attention_dropout: float = 0.10
    classifier_dropout: float = 0.25
    projection_dropout: float = 0.10
    projection_dim: int = 256
    focal_gamma: float = 1.5
    max_class_weight: float = 7.5
    lambda_l1: float = 0.05          # Level-1 auxiliary loss
    lambda_hcl: float = 0.05         # hierarchy-guided contrastive loss
    temperature: float = 0.10
    positive_weight: float = 1.0
    sibling_negative_weight: float = 1.25
    top_k_parents: int = 3           # decoder: top-k parent routing


class HierarchyGuidedSupConLoss(nn.Module):
    def __init__(self, temperature: float, positive_weight: float, sibling_negative_weight: float, eps: float = 1e-12):
        super().__init__()
        self.t, self.pw, self.sw, self.eps = temperature, positive_weight, sibling_negative_weight, eps

    def forward(self, features, l2, l1):
        n = features.size(0)
        if n <= 1:
            return features.new_tensor(0.0)
        z = F.normalize(features, p=2, dim=-1)
        sim = z @ z.T / self.t
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        not_self = ~torch.eye(n, dtype=torch.bool, device=features.device)
        same_l2 = l2.unsqueeze(0) == l2.unsqueeze(1)
        same_l1 = l1.unsqueeze(0) == l1.unsqueeze(1)
        pos = (same_l2 & not_self).float() * self.pw
        sibling = same_l1 & ~same_l2 & not_self
        denom_w = (1.0 + sibling.float() * (self.sw - 1.0)) * not_self.float()
        log_prob = sim - torch.log((torch.exp(sim) * denom_w).sum(dim=1, keepdim=True) + self.eps)
        pos_sum = pos.sum(dim=1)
        valid = pos_sum > 0
        if not valid.any():
            return features.new_tensor(0.0)
        return (-(pos * log_prob).sum(dim=1) / (pos_sum + self.eps))[valid].mean()


class HGCLRModel(nn.Module):
    def __init__(self, cfg: HGCLRConfig, num_l1: int, l2_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(cfg.model_name, num_labels=len(l2_labels))
        config.hidden_dropout_prob = cfg.encoder_dropout
        config.attention_probs_dropout_prob = cfg.attention_dropout
        config.classifier_dropout = cfg.classifier_dropout
        self.seq_model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
        self.config = self.seq_model.config
        self.config.keys_to_ignore_at_inference = ["features"]
        h = config.hidden_size

        def head(drop, out):
            return nn.Sequential(nn.Dropout(drop), nn.Linear(h, h), nn.Tanh(), nn.Dropout(drop), nn.Linear(h, out))

        self.l1_head = head(cfg.classifier_dropout, num_l1)
        self.projection = head(cfg.projection_dropout, cfg.projection_dim)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        seq = self.seq_model.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = seq[:, 0, :]
        return {"logits": self.seq_model.classifier(seq), "l1_logits": self.l1_head(cls), "features": self.projection(cls)}


class HGCLRTrainer(Trainer):
    def __init__(self, *args, focal: FocalLoss, hcl: HierarchyGuidedSupConLoss, lambda_l1: float, lambda_hcl: float, **kw):
        super().__init__(*args, **kw)
        self.focal, self.hcl, self.lambda_l1, self.lambda_hcl = focal, hcl, lambda_l1, lambda_hcl

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        l2, l1 = inputs.pop("labels"), inputs.pop("l1_labels")
        out = model(**inputs)
        loss = self.focal(out["logits"], l2)
        loss = loss + self.lambda_l1 * F.cross_entropy(out["l1_logits"], l1)
        loss = loss + self.lambda_hcl * self.hcl(out["features"], l2, l1)
        return (loss, out) if return_outputs else loss


def top_k_parent_decode(l2_logits: np.ndarray, l1_logits: np.ndarray, parent_of_l2: np.ndarray, k: int) -> np.ndarray:
    """Keep only Level-2 labels whose parent is among the k best parents, then take the argmax."""
    if k >= l1_logits.shape[1]:
        return l2_logits.argmax(axis=1)
    top = np.argsort(-l1_logits, axis=1)[:, :k]
    allowed = np.stack([np.isin(parent_of_l2, top[i]) for i in range(len(top))])
    return np.where(allowed, l2_logits.astype(np.float64), -1e9).argmax(axis=1)


def _macro_f1(eval_pred):
    preds = eval_pred.predictions
    logits = preds[0] if isinstance(preds, (tuple, list)) else preds
    labels = eval_pred.label_ids[0] if isinstance(eval_pred.label_ids, (tuple, list)) else eval_pred.label_ids
    return {"macro_f1": f1_score(labels, np.argmax(logits, axis=1), average="macro", zero_division=0)}


def run(df: pd.DataFrame, l2_labels, l1_labels, l2_to_l1, seed: int, out_dir: str, cfg: HGCLRConfig = HGCLRConfig()) -> list:
    l2_id = {l: i for i, l in enumerate(l2_labels)}
    l1_id = {l: i for i, l in enumerate(l1_labels)}
    parent_of_l2 = np.array([l1_id[l2_to_l1[l]] for l in l2_labels])
    parts = []
    for k in range(N_FOLDS):
        seed_everything(seed)
        train, val, test = fold_split(df, k)
        tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        ds = {name: TextDataset(d["text"], d["L2"].map(l2_id), tok, cfg.max_len, d["L1"].map(l1_id))
              for name, d in (("train", train), ("val", val), ("test", test))}
        run_dir = os.path.join(out_dir, f"_ckpt_seed{seed}_fold{k}")
        shutil.rmtree(run_dir, ignore_errors=True)
        weights = balanced_class_weights(train["L2"].map(l2_id), len(l2_labels), cfg.max_class_weight)
        trainer = HGCLRTrainer(
            model=HGCLRModel(cfg, len(l1_labels), l2_labels),
            args=training_args(run_dir, seed, cfg.batch_size, cfg.lr, cfg.weight_decay, cfg.warmup_ratio, len(train),
                               cfg.max_epochs, extra=dict(remove_unused_columns=False, label_names=["labels", "l1_labels"])),
            train_dataset=ds["train"], eval_dataset=ds["val"], compute_metrics=_macro_f1,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience), FixLayerNormKeys()],
            focal=FocalLoss(alpha=weights, gamma=cfg.focal_gamma),
            hcl=HierarchyGuidedSupConLoss(cfg.temperature, cfg.positive_weight, cfg.sibling_negative_weight),
            lambda_l1=cfg.lambda_l1, lambda_hcl=cfg.lambda_hcl,
        )
        trainer.train()
        l2_logits, l1_logits = trainer.predict(ds["test"]).predictions[:2]
        pred = top_k_parent_decode(np.asarray(l2_logits), np.asarray(l1_logits), parent_of_l2, cfg.top_k_parents)
        parts.append(prediction_frame(test, [l2_labels[i] for i in pred], l2_to_l1, seed))
        shutil.rmtree(run_dir, ignore_errors=True)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return parts
