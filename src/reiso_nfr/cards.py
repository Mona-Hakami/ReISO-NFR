"""ISO/IEC 25010 label cards.

* ``label_cards/retrieval_cards.csv``: full contrastive cards, encoded by Qwen3 in Stage 1.
* ``label_cards/reranker_cards.csv``: short definition + signals cards, read by the
  DeBERTa cross-encoder in Stage 2. It also holds the ISO definition of each label,
  used by the "ISO definitions only" ablation.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd

from .data import ROOT

CARD_DIR = os.path.join(ROOT, "label_cards")


def _definition_only(row) -> str:
    """Card with only the label name, parent name and ISO definition (ablation)."""
    return (f"Quality sub-characteristic: {row['L2_name']}. Parent quality characteristic: {row['L1_name']}. "
            f"Definition: {row['iso_definition']}.")


def load_cards(l2_labels, definitions_only: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (retrieval_card, reranker_card) maps keyed by Level-2 label id."""
    short = pd.read_csv(os.path.join(CARD_DIR, "reranker_cards.csv")).set_index("L2")
    if definitions_only:
        if short["iso_definition"].isna().any():
            raise ValueError("every label needs an ISO definition for this ablation")
        cards = {l2: _definition_only(short.loc[l2]) for l2 in l2_labels}
        return cards, cards
    full = pd.read_csv(os.path.join(CARD_DIR, "retrieval_cards.csv")).set_index("L2")
    missing = (set(l2_labels) - set(full.index)) | (set(l2_labels) - set(short.index))
    if missing:
        raise ValueError(f"labels without a card: {sorted(missing)}")
    return ({l2: full.loc[l2, "full_contrastive_label_card"] for l2 in l2_labels},
            {l2: short.loc[l2, "label_card"] for l2 in l2_labels})
