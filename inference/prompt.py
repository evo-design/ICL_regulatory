"""Compile ICL example sets from a FASTA pool.

Evo 2 does in-context learning by being shown a run of example sequences and
asked to continue the pattern.  This script compiles examples that go into
each prompt and writes one FASTA per set; `Evo2_ICL.py` then turns each set
into a prompt and samples from it.

Three ways to choose, via --mode:

  top_n    one set, the first k records of the pool in file order.  Use this
           when the pool is already ranked (e.g. sorted by measured activity),
           so "the first k" means "the k best".  

  nested   --n_sets independent sets drawn from the top --candidate_top of the
           pool.  The sets are NESTED: the k-shot prompt is the first k of a
           set, so one draw serves every shot count and the shot axis is not
           confounded with a change of examples. Useful for comparing if more 
           examples help. 

  random   --n_sets sets drawn uniformly from the whole pool, unranked. Useful
           for comparing post-trained models with base models. 

The draw is a hash-ordered permutation, not an RNG call: candidate indices are
sorted by sha256(seed | label | set | index).  That depends on nothing but
hashlib, so every machine and every Python build reproduces the same sets, and
a set can be re-derived later to audit what was actually generated from.

    # one ranked prompt per cell type
    python prompt.py --fasta_dir data/human_promoter --mode top_n \\
        --set_size 128 --out out/prompts/human_promoter

    # 32 nested sets from the top 300, for a shot sweep
    python prompt.py --fasta data/5UTR/5UTR.fasta --mode nested \\
        --n_sets 32 --candidate_top 300 --set_size 128 --out out/prompts/5utr

Then point the generator at the result:

    python Evo2_ICL.py --fasta_dir out/prompts/5utr --fasta_suffix .fasta \\
        --fixed_examples --n_shots 1 2 4 8 16 ...
"""

import argparse
import glob
import hashlib
import json
import os


def load_fasta(path):
    """Sequences only, one per record; headers are ignored."""
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif line:
                cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def _hash_key(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def draw(n_candidates, size, seed, label, set_idx):
    """Indices for one set, in the order they should appear in the prompt.

    Sorting the candidates by a hash of (seed, label, set, index) is a
    uniformly random permutation that is reproducible from nothing but
    hashlib.  Taking the first `size` of it makes the sets nested for free:
    the k-shot prompt is the first k entries.
    """
    order = sorted(range(n_candidates),
                   key=lambda i: _hash_key(seed, label, set_idx, i))
    return order[:size]


def build_sets(pool, mode, set_size, n_sets, candidate_top, seed, label):
    """-> [(set_index, [pool indices])].  Pool order is meaningful for top_n."""
    size = min(set_size, len(pool))
    if mode == "top_n":
        return [(0, list(range(size)))]
    n_cand = len(pool) if mode == "random" else min(candidate_top, len(pool))
    if size > n_cand:
        raise SystemExit(
            f"{label}: set_size {size} exceeds the {n_cand} candidates "
            f"available ({mode} mode); lower --set_size or raise "
            f"--candidate_top.")
    return [(j, draw(n_cand, size, seed, label, j)) for j in range(n_sets)]


def write_sets(pool_path, outdir, mode, set_size, n_sets, candidate_top,
               seed, overwrite=False):
    label = os.path.splitext(os.path.basename(pool_path))[0]
    pool = load_fasta(pool_path)
    if not pool:
        raise SystemExit(f"{pool_path}: no sequences")
    os.makedirs(outdir, exist_ok=True)

    written = []
    for set_idx, idx in build_sets(pool, mode, set_size, n_sets,
                                   candidate_top, seed, label):
        name = label if mode == "top_n" else f"{label}_set{set_idx:02d}"
        path = os.path.join(outdir, name + ".fasta")
        if os.path.exists(path) and not overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite")
        seqs = [pool[i] for i in idx]
        with open(path, "w") as f:
            for i, s in zip(idx, seqs):
                # the pool index makes a set traceable back to its source
                # FASTA without re-running the draw
                f.write(f">{name}|pool_index={i}\n{s}\n")
        written.append(dict(
            label=name, cell=label, set_index=set_idx,
            pool=os.path.basename(pool_path), pool_size=len(pool),
            candidates=len(pool) if mode in ("top_n", "random")
                       else min(candidate_top, len(pool)),
            size=len(idx), pool_indices=idx,
            sha256=hashlib.sha256("".join(seqs).encode()).hexdigest()))
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Compile ICL example sets from a FASTA pool.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="a single pool")
    src.add_argument("--fasta_dir", help="one pool per file (e.g. per cell type)")
    ap.add_argument("--fasta_suffix", default=".fasta",
                    help="with --fasta_dir (default: .fasta)")
    ap.add_argument("--mode", required=True,
                    choices=["top_n", "nested", "random"])
    ap.add_argument("--set_size", type=int, required=True,
                    help="examples per set; must cover your largest n_shots")
    ap.add_argument("--n_sets", type=int, default=32,
                    help="nested/random only (default: 32)")
    ap.add_argument("--candidate_top", type=int, default=300,
                    help="nested only: draw from the pool's first N "
                         "(default: 300)")
    ap.add_argument("--seed", type=int, default=20261012,
                    help="hash seed for the draw; same seed -> same sets")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.fasta:
        pools = [args.fasta]
    else:
        pools = sorted(glob.glob(os.path.join(args.fasta_dir,
                                              "*" + args.fasta_suffix)))
        if not pools:
            raise SystemExit(f"no *{args.fasta_suffix} in {args.fasta_dir}")

    manifest = dict(mode=args.mode, set_size=args.set_size,
                    n_sets=1 if args.mode == "top_n" else args.n_sets,
                    candidate_top=args.candidate_top, seed=args.seed, sets=[])
    for p in pools:
        manifest["sets"] += write_sets(
            p, args.out, args.mode, args.set_size,
            1 if args.mode == "top_n" else args.n_sets,
            args.candidate_top, args.seed, overwrite=args.overwrite)

    mpath = os.path.join(args.out, "prompt_sets.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    n = len(manifest["sets"])
    print(f"[prompt] {args.mode}: wrote {n} set(s) from {len(pools)} pool(s) "
          f"-> {args.out}")
    print(f"[prompt] manifest: {mpath}")
    print(f"[prompt] feed them to the generator with:\n"
          f"    python Evo2_ICL.py --fasta_dir {args.out} "
          f"--fasta_suffix .fasta --fixed_examples ...")


if __name__ == "__main__":
    main()
