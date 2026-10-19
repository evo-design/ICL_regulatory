"""generated sequences -> filtered, scored table.

Runs in the task's scoring environment (it imports an oracle).  Reads the
pickles `Evo2_ICL.py` wrote, drops invalid sequences, computes the novelty
metrics, applies the task's filter stack, scores what survives, and writes one
CSV.

    python -m example_scripts.score --gen_dir out/5utr/gen \\
        --out out/5utr/5utr_scored.csv --task_json out/5utr/task.json
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import filters as F


def load_generated(gen_dir):
    """Every {label}_shot{N}.pkl in gen_dir -> one long DataFrame."""
    paths = sorted(p for p in glob.glob(os.path.join(gen_dir, "*.pkl"))
                   if os.path.basename(p) != "all_results.pkl")
    if not paths:
        raise SystemExit(f"no generation pickles in {gen_dir}")
    rows = []
    for p in paths:
        d = pickle.load(open(p, "rb"))
        label = d["label"]
        # "<cell>_set07" -> cell "<cell>", set 7;  "<cell>" -> set 0
        cell, _, tail = label.partition("_set")
        rows += [dict(seq=s, label=label, cell_type=cell,
                      set_index=int(tail) if tail else 0,
                      n_shots=d["n_shots"],
                      examples_file=d["examples_file"])
                 for s in d["sequences"]]
    df = pd.DataFrame(rows)
    print(f"[score] loaded {len(df)} sequences from {len(paths)} file(s)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_jobs", type=int, default=16)
    args = ap.parse_args()

    task = json.load(open(args.task_json))
    df = load_generated(args.gen_dir)

    # 1. drop anything that is not a clean ACGT string of the right length
    ok = df["seq"].apply(F.valid_sequence)
    print(f"[score] {int((~ok).sum())} invalid sequence(s) dropped")
    df = df[ok].reset_index(drop=True)

    # 2. metrics the filters need: GC, and novelty against the prompt's own
    #    examples (identity to the closest example, best local alignment)
    df["GC_ratio"] = df["seq"].apply(F.get_GC_content)
    pools = {f: F.load_examples_from_fasta(f)
             for f in df["examples_file"].unique()}
    pairs = [(s, pools[f][:k])
             for s, f, k in zip(df["seq"], df["examples_file"], df["n_shots"])]
    print(f"[score] novelty for {len(pairs)} sequences "
          f"({args.n_jobs} workers)...")
    nov = F.novelty_many(pairs, n_jobs=args.n_jobs)
    df["sequence_identity"] = [a for a, _ in nov]
    df["blastn_score"] = [b for _, b in nov]

    # 3. the task's filter stack
    df = F.add_filter_columns(df, task)
    df["keep"] = F.keep_mask(df, task)
    kept = int(df["keep"].sum())
    print(f"[score] {kept}/{len(df)} passed the filters "
          f"({100.0 * kept / max(len(df), 1):.1f}%)")

    # 4. score the survivors with the task's oracle
    sub = df[df["keep"]].reset_index(drop=True)
    if len(sub):
        import oracles
        print(f"[score] scoring {len(sub)} with {task['scorer']}...")
        if task["scorer"] == "malinois":
            cwd = oracles.malinois_safe_cwd()
            print(f"[score] chdir -> {cwd} (boda unpacks ./artifacts)")
        sub["score"] = oracles.score(task["scorer"], sub, device=args.device,
                                     seq_len=task["gen_length"])
        thr = task["selected_threshold"]
        sub["selected"] = sub["score"] > thr
        print(f"[score] {int(sub['selected'].sum())}/{len(sub)} above the "
              f"selection threshold ({thr})")
    else:
        sub["score"] = np.array([], dtype=float)
        sub["selected"] = np.array([], dtype=bool)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"[score] wrote {len(sub)} rows -> {args.out}")

    if len(sub):
        print("\n[score] median score by shot count:")
        for k, g in sub.groupby("n_shots"):
            print(f"    {k:>4} shots  n={len(g):<6} "
                  f"median={g['score'].median():.3f}  "
                  f"selected={100.0 * g['selected'].mean():.1f}%")


if __name__ == "__main__":
    main()
