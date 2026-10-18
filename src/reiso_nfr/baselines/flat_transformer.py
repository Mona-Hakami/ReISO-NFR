"""Flat transformer baseline: one softmax over the 24 Level-2 labels.

Fine-tuned RoBERTa-base (main baseline) or DeBERTa-v3-base (robustness check) with
class-weighted focal loss. The best epoch is chosen on the validation fold.
"""
from __future__ import annotations

import gc
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer)

from ..common_torch import FixLayerNormKeys, FocalLoss, TextDataset, balanced_class_weights, training_args
from ..data import N_FOLDS, fold_split, prediction_frame, seed_everything



@dataclass
class FlatConfig:
    model_name: str = "roberta-base"
    max_len: int = 128
    batch_size: int = 8
    lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.08
    max_epochs: int = 30
    patience: int = 3
    hidden_dropout: float = 0.20
    attention_dropout: float = 0.20
    classifier_dropout: float = 0.25
    focal_gamma: float = 1.5
    max_class_weight: float = 7.5


# Settings recorded in the fold-result files of the runs reported in the paper.
CONFIGS = {
    "roberta": FlatConfig(model_name="roberta-base"),
    "deberta": FlatConfig(model_name="microsoft/deberta-v3-base", hidden_dropout=0.15, attention_dropout=0.10),
}


class FocalTrainer(Trainer):
    def __init__(self, *args, focal: FocalLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal = focal

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _macro_f1(eval_pred):
    logits, labels = eval_pred
    return {"macro_f1": f1_score(labels, np.argmax(logits, axis=1), average="macro", zero_division=0)}


def run(df: pd.DataFrame, l2_labels, l2_to_l1, seed: int, out_dir: str, cfg: FlatConfig = FlatConfig()) -> list:
    lab2id = {l: i for i, l in enumerate(l2_labels)}
    parts = []
    for k in range(N_FOLDS):
        seed_everything(seed)
        train, val, test = fold_split(df, k)
        tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        ds = {name: TextDataset(d["text"], d["L2"].map(lab2id), tok, cfg.max_len)
              for name, d in (("train", train), ("val", val), ("test", test))}
        config = AutoConfig.from_pretrained(
            cfg.model_name, num_labels=len(l2_labels), id2label=dict(enumerate(l2_labels)), label2id=lab2id,
            hidden_dropout_prob=cfg.hidden_dropout, attention_probs_dropout_prob=cfg.attention_dropout,
            classifier_dropout=cfg.classifier_dropout,
        )
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
        run_dir = os.path.join(out_dir, f"_ckpt_seed{seed}_fold{k}")
        shutil.rmtree(run_dir, ignore_errors=True)
        weights = balanced_class_weights(train["L2"].map(lab2id), len(l2_labels), cfg.max_class_weight)
        trainer = FocalTrainer(
            model=model,
            args=training_args(run_dir, seed, cfg.batch_size, cfg.lr, cfg.weight_decay, cfg.warmup_ratio,
                               len(train), cfg.max_epochs),
            train_dataset=ds["train"], eval_dataset=ds["val"], compute_metrics=_macro_f1,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience), FixLayerNormKeys()],
            focal=FocalLoss(alpha=weights, gamma=cfg.focal_gamma),
        )
        trainer.train()
        pred = np.argmax(trainer.predict(ds["test"]).predictions, axis=1)
        parts.append(prediction_frame(test, [l2_labels[i] for i in pred], l2_to_l1, seed))
        shutil.rmtree(run_dir, ignore_errors=True)
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return parts
