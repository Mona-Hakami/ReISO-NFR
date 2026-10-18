"""Recompute the result tables of the paper from the saved out-of-fold predictions.

python evaluate.py                # reads results/<system>/seed<seed>.csv, writes results/tables/*.csv

Tables: overall comparison (Table 6), paired tests (Table 7), seed stability (Table 8),
error decomposition (Table 9), per-parent results (Table 10), per-class F1 with 95%
intervals (Table 11), reranker gain over Stage 1, ablations (Table 14) and accuracy by
similarity to the training data (Table 15).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from reiso_nfr.data import N_FOLDS, ROOT, SEEDS, label_info, load_dataset  # noqa: E402
from reiso_nfr.metrics import compute_metrics  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "tables")
SYSTEMS = {  # folder -> name in the paper
    "char_tfidf_svm": "Char TF-IDF + LinearSVM",
    "dbgat": "DBGAT (re-implemented)",
    "flat_roberta": "Flat Transformer (RoBERTa)",
    "flat_deberta": "Flat Transformer (DeBERTa-v3)",
    "iso_hgclr": "ISO-HGCLR: Top-k=3",
    "reiso_nfr": "ReISO-NFR (ISO+kNN)",
}
MAIN = "reiso_nfr"
ABLATIONS = ["pointwise", "no_margin", "no_hard_negatives", "iso_only", "knn_only", "iso_definitions_only"]
DETAIL_SEED = 42
B = 2000  # bootstrap resamples

DF = load_dataset()
L2, L1, PARENT = label_info(DF)
L2I = {l: i for i, l in enumerate(L2)}
GOLD = DF["L2"].map(L2I).values


def load(folder: str, seed: int) -> pd.DataFrame:
    path = os.path.join(RESULTS, folder, f"seed{seed}.csv")
    if not os.path.exists(path):  # the SVM baseline was run once: it has no random initialisation
        path = os.path.join(RESULTS, folder, f"seed{DETAIL_SEED}.csv")
    d = pd.read_csv(path).set_index("id").loc[DF["id"]].reset_index()
    assert (d["gold_L2"].values == DF["L2"].values).all() and (d["test_fold"].values == DF["test_fold"].values).all()
    return d


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def macro_f1(t, p, idx):
    """Fast macro-F1 over the labels present in a (resampled) set."""
    tt, pp, n = t[idx], p[idx], len(L2)
    tp = np.bincount(tt[tt == pp], minlength=n)
    denom = np.bincount(tt, minlength=n) + np.bincount(pp, minlength=n)
    return (2 * tp[denom > 0] / denom[denom > 0]).mean()


def per_class_f1(t, p):
    n = len(L2)
    tp = np.bincount(t[t == p], minlength=n)
    denom = np.bincount(t, minlength=n) + np.bincount(p, minlength=n)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)


def stratified_bootstrap(seed: int):
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(GOLD == c) for c in range(len(L2))]
    return [np.concatenate([rng.choice(pl, len(pl), replace=True) for pl in pools]) for _ in range(B)]


BOOT = stratified_bootstrap(20261014)


def paired(pred_a, pred_b):
    """Paired bootstrap on the macro-F1 difference and McNemar's exact test."""
    a, b = pred_a.map(L2I).values, pred_b.map(L2I).values
    full = np.arange(len(GOLD))
    ds = np.array([macro_f1(GOLD, a, i) - macro_f1(GOLD, b, i) for i in BOOT])
    ca, cb = a == GOLD, b == GOLD
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"delta_pp": 100 * (macro_f1(GOLD, a, full) - macro_f1(GOLD, b, full)),
            "ci_low_pp": 100 * np.percentile(ds, 2.5), "ci_high_pp": 100 * np.percentile(ds, 97.5),
            "p_bootstrap": min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())),
            "a_only_correct": n01, "b_only_correct": n10,
            "p_mcnemar": stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0}


def fold_scores(d):
    return [compute_metrics(g["gold_L2"], g["pred_L2"], PARENT) for _, g in d.groupby("test_fold")]


def corrected_t(diff, n_train, n_test):
    """Corrected resampled t-test (Nadeau and Bengio) over fold-seed differences."""
    diff = np.asarray(diff)
    v = np.var(diff, ddof=1)
    t = diff.mean() / np.sqrt((1 / len(diff) + n_test / n_train) * v) if v > 0 else np.inf
    return t, 2 * stats.t.sf(abs(t), len(diff) - 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {(f, s): load(f, s) for f in SYSTEMS for s in SEEDS}

    # Table 6 and Table 8: fold-mean metrics per seed, then mean and SD over seeds.
    rows = []
    for (f, s), d in runs.items():
        for k, m in enumerate(fold_scores(d)):
            rows.append({"system": SYSTEMS[f], "folder": f, "seed": s, "fold": k, **m})
    folds = pd.DataFrame(rows)
    keys = [c for c in folds.columns if c not in ("system", "folder", "seed", "fold")]
    per_seed = folds.groupby(["system", "seed"])[keys].mean().reset_index()
    overall = per_seed.groupby("system")[keys].agg(["mean", "std"]).reindex(SYSTEMS.values())
    overall.columns = [f"{a}_{b}" for a, b in overall.columns]
    overall.to_csv(os.path.join(OUT, "table6_overall.csv"))
    per_seed.pivot(index="system", columns="seed", values="l2_macro_f1").reindex(SYSTEMS.values()).to_csv(
        os.path.join(OUT, "table8_seed_stability.csv"))
    print(overall[["l2_macro_f1_mean", "l2_macro_f1_std", "l1_macro_f1_mean", "hF1_mean"]].round(3), "\n")

    # Table 9: error decomposition (mean over seeds).
    overall[["correct_l2_pct_mean", "wrong_sibling_pct_mean", "wrong_parent_pct_mean"]].to_csv(
        os.path.join(OUT, "table9_error_decomposition.csv"))

    # Table 7: ReISO-NFR against each baseline; Holm within each seed.
    n_test = len(DF) / N_FOLDS                 # average test-fold size
    n_train = len(DF) * (N_FOLDS - 2) / N_FOLDS  # three training folds
    tests = []
    for s in SEEDS:
        block = [{"baseline": SYSTEMS[f], "seed": s, **paired(runs[(MAIN, s)]["pred_L2"], runs[(f, s)]["pred_L2"])}
                 for f in SYSTEMS if f != MAIN]
        for col in ("p_bootstrap", "p_mcnemar"):
            for r, p in zip(block, holm([r[col] for r in block])):
                r[col + "_holm"] = p
        tests += block
    tests = pd.DataFrame(tests)
    ct = []
    a = folds[folds.folder == MAIN].sort_values(["seed", "fold"])["l2_macro_f1"].values
    for f in SYSTEMS:
        if f == MAIN:
            continue
        b = folds[folds.folder == f].sort_values(["seed", "fold"])["l2_macro_f1"].values
        t, p = corrected_t(a - b, n_train, n_test)
        ct.append({"baseline": SYSTEMS[f], "folds_won": f"{int((a > b).sum())}/{len(a)}", "corrected_t": t, "p_corrected_t": p})
    ct = pd.DataFrame(ct)
    ct["p_corrected_t_holm"] = holm(ct["p_corrected_t"])
    tests.to_csv(os.path.join(OUT, "table7_paired_tests_by_seed.csv"), index=False)
    ct.to_csv(os.path.join(OUT, "table7_corrected_t.csv"), index=False)
    print(tests.groupby("baseline")[["delta_pp", "p_bootstrap_holm"]].agg(["mean", "max"]).round(4), "\n")

    # Reranker gain over Stage 1 alone (the frozen-retriever kNN classifier).
    gain = []
    for s in SEEDS:
        stage1 = load("stage1_only", s)
        fold_delta = 100 * (pd.DataFrame(fold_scores(runs[(MAIN, s)]))["l2_macro_f1"].mean()
                            - pd.DataFrame(fold_scores(stage1))["l2_macro_f1"].mean())
        gain.append({"seed": s, "fold_mean_delta_pp": fold_delta, **paired(runs[(MAIN, s)]["pred_L2"], stage1["pred_L2"])})
    gain = pd.DataFrame(gain)
    gain.to_csv(os.path.join(OUT, "reranker_gain_over_stage1.csv"), index=False)
    print("Reranker gain over Stage 1 (fold mean, pp):", gain["fold_mean_delta_pp"].round(1).tolist(), "\n")

    # Table 10: parent F1 and within-parent macro-F1 (seed 42, mean over folds).
    pp = []
    for f in SYSTEMS:
        d = runs[(f, DETAIL_SEED)]
        for par in L1:
            kids = [l for l in L2 if PARENT[l] == par]
            vals = []
            for _, g in d.groupby("test_fold"):
                inside = g[g["gold_L1"] == par]
                vals.append((f1_score(g["gold_L1"] == par, g["pred_L1"] == par, zero_division=0),
                             f1_score(inside["gold_L2"], inside["pred_L2"], labels=kids, average="macro", zero_division=0)))
            pp.append({"system": SYSTEMS[f], "parent": par, "parent_f1": np.mean([v[0] for v in vals]),
                       "within_parent_macro_f1": np.mean([v[1] for v in vals])})
    pd.DataFrame(pp).to_csv(os.path.join(OUT, "table10_per_parent.csv"), index=False)

    # Table 11: per-class F1 of ReISO-NFR with 95% intervals, and differences (seed 42, pooled OOF).
    boot = stratified_bootstrap(20261015)
    pred = {f: runs[(f, DETAIL_SEED)]["pred_L2"].map(L2I).values for f in ("reiso_nfr", "flat_roberta", "iso_hgclr")}
    f1 = {f: per_class_f1(GOLD, p) for f, p in pred.items()}
    bf1 = {f: np.array([per_class_f1(GOLD[i], p[i]) for i in boot]) for f, p in pred.items()}
    pc = []
    for c, lab in enumerate(L2):
        r = {"parent": PARENT[lab], "class": lab, "n": int((GOLD == c).sum()), "f1": f1[MAIN][c]}
        r["f1_low"], r["f1_high"] = np.nanpercentile(bf1[MAIN][:, c], [2.5, 97.5])
        for b in ("flat_roberta", "iso_hgclr"):
            r[f"delta_vs_{b}"] = f1[MAIN][c] - f1[b][c]
            r[f"delta_vs_{b}_low"], r[f"delta_vs_{b}_high"] = np.nanpercentile(bf1[MAIN][:, c] - bf1[b][:, c], [2.5, 97.5])
        pc.append(r)
    pd.DataFrame(pc).sort_values(["parent", "class"]).to_csv(os.path.join(OUT, "table11_per_class_ci.csv"), index=False)

    # Table 14: ablations against the full model (seed 42).
    full = runs[(MAIN, DETAIL_SEED)]
    ab = [{"variant": "full", **pd.DataFrame(fold_scores(full)).mean().to_dict()}]
    for v in ABLATIONS:
        d = load(f"ablation_{v}", DETAIL_SEED)
        ab.append({"variant": v, **pd.DataFrame(fold_scores(d)).mean().to_dict(), **paired(full["pred_L2"], d["pred_L2"])})
    ab = pd.DataFrame(ab)
    for col in ("p_bootstrap", "p_mcnemar"):
        ab.loc[1:, col + "_holm"] = holm(ab.loc[1:, col])
    ab.to_csv(os.path.join(OUT, "table14_ablation.csv"), index=False)
    print(ab[["variant", "l2_macro_f1", "p_bootstrap_holm", "p_mcnemar_holm"]].round(4), "\n")

    # Table 15: accuracy by TF-IDF similarity of each test requirement to its nearest training requirement.
    x = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(DF["text"].str.lower())
    nn = np.zeros(len(DF))
    for k in range(N_FOLDS):
        te = np.flatnonzero(DF["test_fold"] == k)
        tr = np.flatnonzero(~DF["test_fold"].isin([k, (k + 1) % N_FOLDS]))
        nn[te] = (x[te] @ x[tr].T).max(axis=1).toarray().ravel()
    bins = pd.cut(nn, [0, 0.2, 0.4, 0.6, 1.01], right=False, labels=["<0.2", "0.2-0.4", "0.4-0.6", ">=0.6"])
    sim = pd.DataFrame({f: [np.mean([(runs[(f, s)]["pred_L2"].values == DF["L2"].values)[bins == b].mean() for s in SEEDS])
                            for b in bins.categories] for f in SYSTEMS}, index=bins.categories).T
    sim.index = [SYSTEMS[f] for f in sim.index]
    sim.loc["n"] = [int((bins == b).sum()) for b in bins.categories]
    sim.to_csv(os.path.join(OUT, "table15_similarity.csv"))
    print(sim.round(3))
    print("\nTables written to", OUT)


if __name__ == "__main__":
    main()
