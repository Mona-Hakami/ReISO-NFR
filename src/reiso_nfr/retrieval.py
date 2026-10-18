"""Stage 1: frozen Qwen3 retrieval of candidate Level-2 labels.

score(label) = alpha * cos(requirement, label card)
             + (1 - alpha) * max cos(requirement, training requirement with that label)
over the K_NN nearest training requirements (Section IV). A label with no neighbour
among the K_NN gets a kNN term of 0. The kNN index holds only the training folds, and a
training requirement is never retrieved as its own neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

QUERY_INSTRUCTION = ("Retrieve the ISO/IEC 25010 quality sub-characteristic or similar labeled software "
                     "requirements that best match this software requirement.")


@dataclass
class RetrievalConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 4
    mode: str = "iso_plus_knn"     # iso_plus_knn | iso_only (alpha = 1) | knn_only (alpha = 0)
    knn_top_n: int = 20            # K_NN
    candidate_k: int = 10          # K_c
    iso_weight: float = 0.5        # alpha
    knn_weight: float = 0.5        # 1 - alpha


def load_retriever(cfg: RetrievalConfig):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(cfg.model_name, device=device, trust_remote_code=True,
                               model_kwargs={"torch_dtype": torch.float16})


def encode(model, texts: List[str], batch_size: int, is_query: bool) -> np.ndarray:
    """L2-normalised embeddings. Requirements used as queries get the Qwen3 instruction;
    label cards and the training corpus are encoded without one."""
    prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: " if is_query else None
    kw = dict(batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    emb = model.encode(list(map(str, texts)), prompt=prompt, **kw) if prompt else model.encode(list(map(str, texts)), **kw)
    return np.asarray(emb, dtype=np.float32)


def rank_labels(query_emb: np.ndarray, train_emb: np.ndarray, label_emb: np.ndarray, label_ids: List[str],
                train_l2: List[str], cfg: RetrievalConfig, self_positions: Optional[List[int]] = None) -> List[List[str]]:
    """Ranked candidate labels for every query (best first)."""
    use_iso = cfg.mode in {"iso_only", "iso_plus_knn"}
    use_knn = cfg.mode in {"knn_only", "iso_plus_knn"}
    iso = query_emb @ label_emb.T
    sims = None
    if use_knn:
        sims = query_emb @ train_emb.T
        if self_positions is not None:  # never retrieve a training requirement as its own neighbour
            sims[np.arange(len(self_positions)), self_positions] = -999.0
    ranked = []
    for i in range(len(query_emb)):
        score: Dict[str, float] = {}
        if use_iso:
            for j, lab in enumerate(label_ids):
                score[lab] = score.get(lab, 0.0) + cfg.iso_weight * float(iso[i, j])
        if use_knn:
            n = min(cfg.knn_top_n, sims.shape[1])
            top = np.argpartition(-sims[i], range(n))[:n]
            top = top[np.argsort(-sims[i, top])]
            best: Dict[str, float] = {}
            for t in top:  # max similarity per label among the K_NN neighbours
                best[train_l2[t]] = max(best.get(train_l2[t], -np.inf), float(sims[i, t]))
            for lab, s in best.items():
                score[lab] = score.get(lab, 0.0) + cfg.knn_weight * s
        ranked.append([lab for lab, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)])
    return ranked


def fold_rankings(model, train, val, test, label_ids, label_emb, cfg: RetrievalConfig) -> Dict[str, List[List[str]]]:
    """Encode one fold and rank labels for its training, validation and test requirements."""
    bs = cfg.batch_size
    train_doc = encode(model, train["text"], bs, is_query=False)
    out = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        q = encode(model, d["text"], bs, is_query=True)
        self_pos = list(range(len(train))) if name == "train" else None
        out[name] = rank_labels(q, train_doc, label_emb, label_ids, train["L2"].tolist(), cfg, self_pos)
    return out
