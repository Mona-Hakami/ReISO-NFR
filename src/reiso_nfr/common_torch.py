"""Pieces shared by the transformer baselines (flat transformer and ISO-HGCLR)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


def balanced_class_weights(label_ids, num_labels: int, max_weight: Optional[float] = 7.5) -> torch.Tensor:
    """Inverse-frequency weights from the training folds, normalised to mean 1 and capped."""
    counts = np.bincount(np.asarray(label_ids, dtype=int), minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_labels * counts)
    w = w / max(w.mean(), 1e-12)
    if max_weight is not None:
        w = np.minimum(w, float(max_weight))
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Class-weighted focal loss (Eq. 11): alpha_y * (1 - p_y)^gamma * (-log p_y)."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_pt = F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1.0 - log_pt.exp()).pow(self.gamma) * (-log_pt)
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        return loss.mean()


class TextDataset(torch.utils.data.Dataset):
    """Tokenised requirements with Level-2 (and optionally Level-1) targets."""

    def __init__(self, texts, l2_ids, tokenizer, max_len: int, l1_ids=None):
        self.texts, self.l2, self.l1 = list(texts), np.asarray(l2_ids), None if l1_ids is None else np.asarray(l1_ids)
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.l2[i], dtype=torch.long)
        if self.l1 is not None:
            item["l1_labels"] = torch.tensor(self.l1[i], dtype=torch.long)
        return item


class FixLayerNormKeys(TrainerCallback):
    """Rename LayerNorm gamma/beta keys in saved checkpoints so that
    ``load_best_model_at_end`` can reload them with recent transformers versions."""

    def on_save(self, args, state, control, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        for name in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(ckpt, name)
            if not os.path.exists(path):
                continue
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file, save_file

                sd = load_file(path)
            else:
                sd = torch.load(path, map_location="cpu")
            if not any(k.endswith(("LayerNorm.gamma", "LayerNorm.beta")) for k in sd):
                continue
            fixed = {k.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias"): v
                     for k, v in sd.items()}
            if name.endswith(".safetensors"):
                save_file(fixed, path)
            else:
                torch.save(fixed, path)
        return control


def training_args(out_dir: str, seed: int, batch_size: int, lr: float, wd: float, warmup_ratio: float,
                  n_train: int, max_epochs: int, extra: Optional[dict] = None):
    """Hugging Face TrainingArguments used by both transformer baselines."""
    import inspect
    import math

    from transformers import TrainingArguments

    steps = math.ceil(n_train / batch_size) * max_epochs
    kw = dict(
        output_dir=out_dir, seed=seed, data_seed=seed,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs, learning_rate=lr, weight_decay=wd,
        warmup_steps=int(warmup_ratio * steps), lr_scheduler_type="linear", optim="adamw_torch",
        max_grad_norm=1.0, save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, report_to="none", fp16=torch.cuda.is_available(), full_determinism=True,
    )
    kw.update(extra or {})
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})
