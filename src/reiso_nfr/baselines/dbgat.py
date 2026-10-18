"""Re-implementation of DBGAT (Li et al., IEEE Access, 2022) for the 24-class task.

Architecture as published: dependency parse (Stanza) -> one graph per requirement
(nodes = words, edges = dependency relations) -> node vectors from bert-base-uncased
(mean of sub-words) -> 3 bidirectional GAT layers (hidden 300, 1 head, input dropout
0.3) -> max pooling -> MLP -> softmax.

Choices the original paper leaves open (Section V-A): Stanza parser; the last four
BERT layers are fine-tuned (lr 2e-5); class-weighted cross-entropy (weights
normalised to mean 1, capped at 7.5); the GAT/MLP learning rate is chosen on the
validation fold from {1e-3, 3e-3}; early stopping with patience 5, at most 50 epochs.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..data import N_FOLDS, fold_split, prediction_frame, seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DBGATConfig:
    bert_name: str = "bert-base-uncased"
    bert_dim: int = 768
    bert_max_len: int = 128
    gat_layers: int = 3
    hidden: int = 300
    heads: int = 1
    feat_dropout: float = 0.3
    batch_size: int = 50
    max_epochs: int = 50
    patience: int = 5
    lr_grid: tuple = (1e-3, 3e-3)
    finetune_last_n: int = 4
    bert_lr: float = 2e-5
    max_class_weight: float = 7.5


def _clean(t: str) -> str:
    t = str(t).replace("​", " ").replace("﻿", " ")
    return re.sub(r"\s+", " ", t).strip()


def parse_dependencies(df: pd.DataFrame, cache_path: str) -> dict:
    """Stanza dependency parses, cached because parsing is slow."""
    if os.path.exists(cache_path):
        parses = json.load(open(cache_path))
        if set(parses) == set(df["id"]):
            return parses
    import stanza

    stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", tokenize_no_ssplit=True,
                          use_gpu=torch.cuda.is_available(), verbose=False)
    docs = nlp([stanza.Document([], text=_clean(t)) for t in df["text"]])
    parses = {}
    for rid, doc in zip(df["id"], docs):
        words, heads, roots, off = [], [], [], 0
        for sent in doc.sentences:
            for w in sent.words:
                words.append(w.text)
                heads.append(-1 if w.head == 0 else off + w.head - 1)
                if w.head == 0:
                    roots.append(off + w.id - 1)
            off += len(sent.words)
        for a, b in zip(roots[:-1], roots[1:]):  # link sentence roots into one graph
            heads[b] = a
        parses[rid] = {"words": words, "heads": heads}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(parses, open(cache_path, "w"))
    return parses


def build_graphs(df, parses, l2_id, cfg: DBGATConfig) -> dict:
    from torch_geometric.data import Data
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.bert_name)
    graphs = {}
    for r in df.itertuples():
        words, heads = parses[r.id]["words"], parses[r.id]["heads"]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=cfg.bert_max_len,
                  padding="max_length", return_tensors="pt")
        s2w = torch.tensor([-1 if w is None else w for w in enc.word_ids()], dtype=torch.long)
        src = [h for h in heads if h >= 0]
        dst = [i for i, h in enumerate(heads) if h >= 0]
        ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        graphs[r.id] = Data(edge_index=ei, num_nodes=len(words), y=torch.tensor([l2_id[r.L2]]),
                            input_ids=enc["input_ids"], attn=enc["attention_mask"], s2w=s2w.unsqueeze(0))
    return graphs


class BiGATLayer(nn.Module):
    """Forward (head -> dependent) and backward neighbours, combined with the previous state."""

    def __init__(self, din, dh, heads):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.fwd = GATConv(din, dh, heads=heads, concat=False)
        self.bwd = GATConv(din, dh, heads=heads, concat=False)
        self.comb = nn.Linear(din + 2 * dh, dh)

    def forward(self, h, ei):
        return F.relu(self.comb(torch.cat([h, self.fwd(h, ei), self.bwd(h, ei.flip(0))], dim=-1)))


class DBGAT(nn.Module):
    def __init__(self, n_cls: int, cfg: DBGATConfig):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(cfg.bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        if cfg.finetune_last_n > 0:
            for layer in self.bert.encoder.layer[-cfg.finetune_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.bf16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
        self.drop = nn.Dropout(cfg.feat_dropout)
        self.layers = nn.ModuleList([BiGATLayer(cfg.bert_dim if i == 0 else cfg.hidden, cfg.hidden, cfg.heads)
                                     for i in range(cfg.gat_layers)])
        self.mlp = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, n_cls))

    def node_features(self, data):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            out = self.bert(input_ids=data.input_ids, attention_mask=data.attn).last_hidden_state
        out = out.float()
        valid = data.s2w >= 0
        gidx = (data.s2w + data.ptr[:-1].unsqueeze(1))[valid]
        feats = torch.zeros(data.num_nodes, out.size(-1), device=out.device).index_add_(0, gidx, out[valid])
        cnt = torch.zeros(data.num_nodes, device=out.device).index_add_(0, gidx, torch.ones_like(gidx, dtype=torch.float))
        empty = cnt == 0  # words cut by truncation: use [CLS]
        if empty.any():
            feats[empty] = out[data.batch[empty], 0]
            cnt[empty] = 1
        return feats / cnt.unsqueeze(1)

    def forward(self, data):
        from torch_geometric.nn import global_max_pool

        h = self.node_features(data)
        for layer in self.layers:
            h = layer(self.drop(h), data.edge_index)
        return self.mlp(global_max_pool(h, data.batch))


@torch.no_grad()
def _predict(model, graphs):
    from torch_geometric.loader import DataLoader

    model.eval()
    return torch.cat([model(b.to(DEVICE)).argmax(-1).cpu() for b in DataLoader(graphs, batch_size=128)]).numpy()


def _train_one(lr, seed, tr_g, va_g, va_y, weights, n_cls, cfg):
    from torch_geometric.loader import DataLoader

    seed_everything(seed)
    model = DBGAT(n_cls, cfg).to(DEVICE)
    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")], "lr": lr}]
    bert_p = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_p:
        groups.append({"params": bert_p, "lr": cfg.bert_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    loader = DataLoader(tr_g, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(b), b.y, weight=weights).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vf1 = f1_score(va_y, _predict(model, va_g), average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    return best_f1, best_state


def run(df: pd.DataFrame, l2_labels, l2_to_l1, seed: int, out_dir: str, cfg: DBGATConfig = DBGATConfig()) -> list:
    l2_id = {l: i for i, l in enumerate(l2_labels)}
    graphs = build_graphs(df, parse_dependencies(df, os.path.join(out_dir, "_cache", "stanza_parses.json")), l2_id, cfg)
    parts = []
    for k in range(N_FOLDS):
        train, val, test = fold_split(df, k)
        tr_g, va_g, te_g = ([graphs[i] for i in d["id"]] for d in (train, val, test))
        cnt = np.bincount(train["L2"].map(l2_id), minlength=len(l2_labels)).astype(float)
        w = len(train) / (len(l2_labels) * np.maximum(cnt, 1))
        w = torch.tensor(np.minimum(w / w.mean(), cfg.max_class_weight), dtype=torch.float, device=DEVICE)
        va_y = val["L2"].map(l2_id).values
        best = max((_train_one(lr, seed, tr_g, va_g, va_y, w, len(l2_labels), cfg) for lr in cfg.lr_grid),
                   key=lambda r: r[0])  # learning rate chosen on the validation fold
        model = DBGAT(len(l2_labels), cfg).to(DEVICE)
        model.load_state_dict(best[1])
        parts.append(prediction_frame(test, [l2_labels[i] for i in _predict(model, te_g)], l2_to_l1, seed))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return parts
