"""Train and evaluate one system on the fixed five folds, for one or more seeds.

Examples
--------
python run.py --model reiso_nfr                       # full model, seeds 42 123 2024
python run.py --model reiso_nfr --variant no_margin --seeds 42
python run.py --model roberta
python run.py --model svm
python run.py --model knn_sensitivity                 # Table 13 (retrieval only, no training)

Out-of-fold predictions are written to results/<system>/seed<seed>.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from reiso_nfr.data import ROOT, SEEDS, load_dataset, label_info, save_oof  # noqa: E402
from reiso_nfr.metrics import fold_mean_metrics  # noqa: E402

MODELS = ["svm", "roberta", "deberta", "hgclr", "dbgat", "reiso_nfr", "knn_sensitivity"]


def report(name: str, path: str, l2_to_l1) -> None:
    import pandas as pd

    m = fold_mean_metrics(pd.read_csv(path), l2_to_l1)
    print(f"{name}: L2 macro-F1 {m['l2_macro_f1']:.4f} | accuracy {m['l2_accuracy']:.4f} | hF1 {m['hF1']:.4f} -> {path}")


def knn_sensitivity(df, l2_labels, out_dir):
    """Table 13: test candidate recall at K_c by evidence source and K_NN.

    The retriever is frozen, so no training and no seed are involved."""
    from dataclasses import replace

    import pandas as pd

    from reiso_nfr.cards import load_cards
    from reiso_nfr.data import N_FOLDS, fold_split
    from reiso_nfr.retrieval import RetrievalConfig, encode, load_retriever, rank_labels

    base = RetrievalConfig()
    model = load_retriever(base)
    cards, _ = load_cards(l2_labels)
    label_emb = encode(model, [cards[l] for l in l2_labels], base.batch_size, is_query=False)
    settings = [("iso_only", 0)] + [("iso_plus_knn", n) for n in (1, 3, 5, 10, 20)] + [("knn_only", 20)]
    rows = []
    for k in range(N_FOLDS):
        train, _, test = fold_split(df, k)
        train_doc = encode(model, train["text"], base.batch_size, is_query=False)
        q = encode(model, test["text"], base.batch_size, is_query=True)
        for mode, knn in settings:
            cfg = replace(base, mode=mode, knn_top_n=max(knn, 1))
            ranked = rank_labels(q, train_doc, label_emb, l2_labels, train["L2"].tolist(), cfg)
            for kc in (1, 3, 5, 10):
                hit = [g in r[:kc] for g, r in zip(test["L2"], ranked)]
                rows.append({"retrieval": mode, "K_NN": knn, "K_c": kc, "fold": k,
                             "recall": sum(hit) / len(hit), "missing": len(hit) - sum(hit)})
    r = pd.DataFrame(rows)
    table = r.groupby(["retrieval", "K_NN", "K_c"]).agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                                                        missing=("missing", "sum")).reset_index()
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "candidate_recall.csv"), index=False)
    print((100 * table.pivot_table(index=["retrieval", "K_NN"], columns="K_c", values="recall_mean")).round(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--variant", default="full", help="ReISO-NFR ablation (see src/reiso_nfr/pipeline.py)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    df = load_dataset()
    l2_labels, l1_labels, l2_to_l1 = label_info(df)

    if args.model == "knn_sensitivity":
        knn_sensitivity(df, l2_labels, os.path.join(args.out, "knn_sensitivity"))
        return

    for seed in args.seeds:
        if args.model == "svm":
            from reiso_nfr.baselines import tfidf_svm

            out = os.path.join(args.out, "char_tfidf_svm")
            parts = tfidf_svm.run(df, l2_to_l1, seed)
        elif args.model in ("roberta", "deberta"):
            from reiso_nfr.baselines import flat_transformer as ft

            out = os.path.join(args.out, f"flat_{args.model}")
            parts = ft.run(df, l2_labels, l2_to_l1, seed, out, ft.CONFIGS[args.model])
        elif args.model == "hgclr":
            from reiso_nfr.baselines import iso_hgclr

            out = os.path.join(args.out, "iso_hgclr")
            parts = iso_hgclr.run(df, l2_labels, l1_labels, l2_to_l1, seed, out)
        elif args.model == "dbgat":
            from reiso_nfr.baselines import dbgat

            out = os.path.join(args.out, "dbgat")
            parts = dbgat.run(df, l2_labels, l2_to_l1, seed, out)
        else:
            from reiso_nfr import pipeline

            name = "reiso_nfr" if args.variant == "full" else f"ablation_{args.variant}"
            out = os.path.join(args.out, name)
            parts, stage1 = pipeline.run(df, l2_labels, l2_to_l1, seed, args.variant)
            if args.variant == "full":
                report("Stage 1 alone", save_oof(stage1, df, os.path.join(args.out, "stage1_only"), seed), l2_to_l1)
        report(f"{os.path.basename(out)} seed {seed}", save_oof(parts, df, out, seed), l2_to_l1)


if __name__ == "__main__":
    main()
