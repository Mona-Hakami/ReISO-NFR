# ReISO-NFR

Code, data and results for the paper **"ReISO-NFR: An ISO/IEC 25010-Guided Retrieval–Reranking Approach for Fine-Grained Non-Functional Requirements Classification"** (M. A. Hakami, M. R. Alrashoud, M. S. Alrazgan).

ReISO-NFR classifies a non-functional requirement into one of 24 ISO/IEC 25010 sub-characteristics in two stages:

1. **Retrieval (frozen).** `Qwen/Qwen3-Embedding-0.6B` scores every label by
   `alpha * cos(requirement, ISO label card) + (1 - alpha) * max cos(requirement, training requirement with that label)`
   over the 20 nearest training requirements (alpha = 0.5), and keeps the top 10 labels.
2. **Reranking (fine-tuned).** A `microsoft/deberta-v3-base` cross-encoder reads the requirement with each candidate's label card and picks the best one. It is trained with a listwise loss, a hierarchy-aware margin loss and sibling hard negatives.

## Repository layout

```
data/dataset_with_test_fold.csv   1,793 labelled requirements and their fixed fold (test_fold)
label_cards/                      ISO/IEC 25010 label cards (retrieval and reranker versions)
src/reiso_nfr/                    data and folds, metrics, retrieval, reranker, baselines
run.py                            train and evaluate one system for one or more seeds
evaluate.py                       recompute the paper's tables from the saved predictions
results/<system>/seed<seed>.csv   out-of-fold predictions of every system and seed
results/tables/                   tables written by evaluate.py
figures/                          figures used in the paper
```

## Setup

```bash
git clone https://github.com/Mona-Hakami/ReISO-NFR.git
cd ReISO-NFR
pip install -r requirements.txt
```

The transformer models need a GPU; the experiments were run on Google Colab Pro with one NVIDIA A100. `notebooks/run_in_colab.ipynb` runs the same commands in Colab.

## Reproduce the tables from the saved predictions

```bash
python evaluate.py
```

This needs no GPU and takes about 10 seconds. It recomputes the overall comparison, the paired tests (paired stratified bootstrap with 2,000 resamples, McNemar's test, corrected resampled t-test, Holm correction), seed stability, error decomposition, per-parent and per-class results with 95% intervals, the reranker gain over Stage 1, the ablations and the similarity analysis.

## Re-run an experiment

```bash
python run.py --model reiso_nfr                        # ReISO-NFR, seeds 42, 123 and 2024 (also writes Stage 1 alone)
python run.py --model reiso_nfr --variant no_margin --seeds 42
python run.py --model roberta                          # also: deberta, hgclr, dbgat, svm
python run.py --model knn_sensitivity                  # candidate recall by K_NN (retrieval only)
```

Ablation variants (`--variant`): `pointwise`, `no_margin`, `no_hard_negatives`, `iso_only` (alpha = 1), `knn_only` (alpha = 0), `iso_definitions_only`. All hyperparameters are in the `*Config` dataclasses in `src/reiso_nfr/`.

## Evaluation protocol

All systems use the same five folds, stratified by Level-2 label and stored in `test_fold`. When fold *k* is the test fold, fold (*k* + 1) mod 5 is the validation fold (early stopping and checkpoint selection on Level-2 macro-F1) and the other three folds are the training set. Every system was trained with seeds 42, 123 and 2024 on the same folds; the seed changes only model training. The character TF-IDF + LinearSVM baseline gives identical predictions for every seed, so it is stored once.

Each prediction file has the columns `id, test_fold, gold_L1, gold_L2, pred_L2, pred_L1, seed` (ReISO-NFR files also list the retrieved `candidates`).

| Folder in `results/` | System in the paper |
|---|---|
| `char_tfidf_svm` | Char TF-IDF + LinearSVM |
| `dbgat` | DBGAT (re-implemented) |
| `flat_roberta`, `flat_deberta` | Flat Transformer (RoBERTa, DeBERTa-v3) |
| `iso_hgclr` | ISO-HGCLR: Top-k=3 |
| `stage1_only` | Stage 1 alone (top-ranked retrieved label) |
| `reiso_nfr` | ReISO-NFR (ISO+kNN) |
| `ablation_*` | Component ablations (seed 42) |
| `knn_sensitivity` | Candidate recall by evidence source and K_NN (retrieval only) |

## Data

`data/dataset_with_test_fold.csv` holds the 1,793 requirements used in the paper. They come from the non-functional requirements of the FR_NFR dataset ([Mendeley Data](https://data.mendeley.com/datasets/4ysx9fyzv4/1)), annotated by three annotators with ISO/IEC 25010 characteristics (Level 1) and sub-characteristics (Level 2). See `data/README.md` for the columns and the exclusion steps.

## Citation

If you use this code or data, please cite the paper (citation details will be added after publication).
