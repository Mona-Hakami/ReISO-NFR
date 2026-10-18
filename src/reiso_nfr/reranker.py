"""Stage 2: listwise DeBERTa-v3 cross-encoder reranker (Section IV).

Each requirement is paired with the label card of each of its K_c candidates; the
cross-encoder gives one score per pair, and the prediction is the highest-scoring
candidate. Training uses
  * the listwise softmax cross-entropy over the candidate list, class-weighted by the
    gold Level-2 label (weights clipped to [1/5, 5]);
  * a hierarchy-aware margin loss: the gold score must exceed each sibling negative by
    m_sib and each cross-parent negative by m_global;
  * up to K_h sibling hard negatives forced into the training candidate lists.
During training the gold label is always in the list; during validation and testing
it is not forced, so a retrieval miss cannot be corrected.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RerankerConfig:
    model_name: str = "microsoft/deberta-v3-base"
    candidate_k: int = 10            # K_c
    max_len: int = 256
    dropout: float = 0.10            # hidden, attention and classifier dropout
    epochs: int = 30
    patience: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    train_batch_size: int = 2
    eval_batch_size: int = 4
    class_weight_clip: float = 5.0   # weights in [1/5, 5]
    loss: str = "listwise"           # listwise | pointwise (ablation)
    hard_negatives: int = 3          # K_h sibling hard negatives; 0 = ablation
    margin_lambda: float = 0.05      # 0 = ablation without margin loss
    margin_sibling: float = 0.20
    margin_global: float = 0.05


class CandidateDataset(Dataset):
    def __init__(self, df, ranked: List[List[str]], cards: Dict[str, str], tokenizer, cfg: RerankerConfig,
                 label_to_parent: Dict[str, str], train: bool, seed: int = 0):
        self.texts, self.gold = df["text"].tolist(), df["L2"].tolist()
        self.ranked, self.cards, self.tok, self.cfg = ranked, cards, tokenizer, cfg
        self.train, self.seed = train, seed
        self.parent = label_to_parent
        self.siblings: Dict[str, List[str]] = {}
        for lab in sorted(cards):
            self.siblings.setdefault(label_to_parent[lab], []).append(lab)

    def __len__(self):
        return len(self.texts)

    def _hard_siblings(self, gold: str, cands: List[str], idx: int) -> List[str]:
        """Siblings already retrieved come first, then other siblings in a fixed random order."""
        pool = [l for l in self.siblings[self.parent[gold]] if l != gold]
        retrieved = [c for c in cands if c in pool]
        rest = [s for s in pool if s not in retrieved]
        random.Random(self.seed + idx).shuffle(rest)
        return (retrieved + rest)[: self.cfg.hard_negatives]

    def _candidates(self, idx: int) -> List[str]:
        k, gold = self.cfg.candidate_k, self.gold[idx]
        cands = list(dict.fromkeys(self.ranked[idx][:k]))
        if self.train:  # force the gold label (and hard siblings) in, dropping the lowest-ranked others
            protected = [gold] + (self._hard_siblings(gold, cands, idx) if self.cfg.hard_negatives > 0 else [])
            cands += [p for p in protected if p not in cands]
            while len(cands) > k:
                drop = next((j for j in range(len(cands) - 1, -1, -1) if cands[j] not in protected), None)
                if drop is None:
                    cands = protected[:k]
                    break
                cands.pop(drop)
        while len(cands) < k:  # pad by repeating the last candidate
            cands.append(cands[-1])
        return cands[:k]

    def __getitem__(self, idx: int):
        cands = self._candidates(idx)
        enc = self.tok([self.texts[idx]] * len(cands), [self.cards[c] for c in cands], truncation=True,
                       padding="max_length", max_length=self.cfg.max_len, return_tensors="pt")
        gold = self.gold[idx]
        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "gold_index": torch.tensor(cands.index(gold) if gold in cands else -1), "gold": gold, "cands": cands}
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("input_ids", "attention_mask", "gold_index")}
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
    out["gold"] = [b["gold"] for b in batch]
    out["cands"] = [b["cands"] for b in batch]
    return out


def class_weights(train_l2: List[str], labels: List[str], clip: float) -> Dict[str, float]:
    counts = {l: train_l2.count(l) for l in labels}
    n, c = len(train_l2), len(labels)
    return {l: float(np.clip(n / (c * counts[l]) if counts[l] else 1.0, 1.0 / clip, clip)) for l in labels}


def _scores(model, batch):
    b, k, length = batch["input_ids"].shape
    inputs = {key: batch[key].to(DEVICE).reshape(b * k, length)
              for key in ("input_ids", "attention_mask", "token_type_ids") if key in batch}
    return model(**inputs).logits.reshape(b, k)


def _margin_loss(scores, batch, parent, cfg: RerankerConfig):
    losses = []
    for i, gi in enumerate(batch["gold_index"].tolist()):
        if gi < 0:
            continue
        gold, seen, item = batch["gold"][i], set(), []
        for j, neg in enumerate(batch["cands"][i]):
            if j == gi or neg == gold or neg in seen:  # skip the gold and repeated padding
                continue
            seen.add(neg)
            m = cfg.margin_sibling if parent[neg] == parent[gold] else cfg.margin_global
            item.append(F.relu(m - scores[i, gi] + scores[i, j]))
        if item:
            losses.append(torch.stack(item).mean())
    return torch.stack(losses).mean() if losses else None


def batch_loss(model, batch, weights, parent, cfg: RerankerConfig):
    scores = _scores(model, batch)
    gold_index = batch["gold_index"].to(DEVICE)
    valid = gold_index >= 0
    if not valid.any():
        return None
    s, g = scores[valid], gold_index[valid]
    if cfg.loss == "listwise":
        per_item = F.cross_entropy(s, g, reduction="none")
    elif cfg.loss == "pointwise":  # ablation: binary cross-entropy per candidate
        target = torch.zeros_like(s)
        target[torch.arange(len(g), device=s.device), g] = 1.0
        rows = [c for c, v in zip(batch["cands"], valid.tolist()) if v]
        mask = torch.tensor([[1.0 if lab not in row[:j] else 0.0 for j, lab in enumerate(row)] for row in rows],
                            dtype=s.dtype, device=s.device)
        per_item = (F.binary_cross_entropy_with_logits(s, target, reduction="none") * mask).sum(1) / mask.sum(1)
    else:
        raise ValueError(f"unknown loss: {cfg.loss}")
    w = torch.tensor([weights[batch["gold"][i]] for i, v in enumerate(valid.tolist()) if v], device=DEVICE)
    loss = (per_item * w).mean()
    if cfg.margin_lambda > 0:
        margin = _margin_loss(scores, batch, parent, cfg)
        if margin is not None:
            loss = loss + cfg.margin_lambda * margin
    return loss


@torch.no_grad()
def predict(model, dataset, cfg: RerankerConfig):
    model.eval()
    preds = []
    for batch in DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate):
        best = _scores(model, batch).argmax(dim=1).tolist()
        preds += [batch["cands"][i][b] for i, b in enumerate(best)]
    return preds


def new_model(cfg: RerankerConfig):
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=1)
    for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "summary_first_dropout"):
        if hasattr(config, attr):
            setattr(config, attr, cfg.dropout)
    return AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config).float()


def train(model, train_ds, val_ds, weights, parent, labels, cfg: RerankerConfig):
    """Train with early stopping on validation Level-2 macro-F1; return the best model."""
    model.to(DEVICE)
    loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(len(loader) * cfg.epochs, 1)
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg.warmup_ratio), total)
    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(cfg.epochs):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, weights, parent, cfg)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        f1 = f1_score(val_ds.gold, predict(model, val_ds, cfg), labels=labels, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model
