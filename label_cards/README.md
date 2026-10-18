# Label cards

One card per ISO/IEC 25010 sub-characteristic (24 labels), derived from the annotation guideline.

| File | Used by | Main column |
|---|---|---|
| `retrieval_cards.csv` | Stage 1, encoded by Qwen3 | `full_contrastive_label_card`: definition, parent definition, typical signals, when to choose and not choose the label, and how it differs from its siblings |
| `reranker_cards.csv` | Stage 2, read by the DeBERTa cross-encoder | `label_card`: label name, parent, ISO definition and typical signals |

`reranker_cards.csv` also holds the ISO/IEC 25010 definition of each label (`iso_definition`), used by the "ISO definitions only" ablation.
