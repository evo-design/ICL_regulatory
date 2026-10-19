"""Generate ICL sequences with Evo 2.

Usage
-----
    export PATH=$CONDA_PREFIX/bin:$PATH        # flashinfer JITs its sampler
    export VLLM_ENABLE_V1_MULTIPROCESSING=0    # real tracebacks

    # one cell type, one shot count, 1 sequence per prompt
    python scripts/generate_icl_vllm.py --model ckpt/evo2_7b_hf \
        --fasta /path/K562_high_activity.fasta \
        --n_shots 16 --gen_length 230 --output_dir results/icl_vllm

    python scripts/generate_icl_vllm.py --model ckpt/evo2_7b_hf \
        --fasta_dir /path/human_promoter --fasta_suffix _high_activity.fasta \
        --n_shots 2 4 8 16 32 --n_seqs 32 --batch_size 32 \
        --gen_length 230 --output_dir results/icl_sweep

    # shapes and token counts only, no GPU
    python scripts/generate_icl_vllm.py --model ckpt/evo2_7b_hf \
        --fasta_dir /path/human_promoter --n_shots 16 --dry_run --output_dir /tmp/x
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

# The Evo 2 vLLM plugin is a separate repository 
# Point $EVO2_VLLM_REPO at a clone; it supplies `vllm_evo2`, which registers
# the Evo 2 architecture with vLLM.
_PLUGIN = os.environ.get("EVO2_VLLM_REPO")
if not _PLUGIN:
    raise SystemExit(
        "$EVO2_VLLM_REPO is not set. Point it at a clone of vllm_evo2 "
        "plugin this script imports.")
if not os.path.isdir(os.path.join(_PLUGIN, "vllm_evo2")):
    raise SystemExit(f"$EVO2_VLLM_REPO={_PLUGIN} has no vllm_evo2/ directory.")
sys.path.insert(0, _PLUGIN)
import vllm_evo2  


# ---------------------------------------------------------------------------
# Data helpers -- identical semantics to rl/generate_icl.py so prompts match
# ---------------------------------------------------------------------------

def load_examples_from_fasta(
    fasta_file: str, min_example_len: int = 1000
) -> list[tuple[str, str]]:
    """Parse a FASTA into one (record_id, sequence) pair per `>` record.

    Every real FASTA is hard-wrapped (60 or 80 columns), so the lines of a record
    must be accumulated: treating each line as its own example turns one 5 kb
    record into ~84 bogus 60-nt "examples" and fills the ICL prompt with fragments.
    A record shorter than `min_example_len` is an error, not a warning -- it almost
    always means the file was parsed or produced wrongly.
    """
    records: list[tuple[str, str]] = []
    parts: list[str] = []
    rec_id: str | None = None

    def flush():
        if rec_id is not None:
            records.append((rec_id, "".join(parts)))

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                rec_id = line[1:].strip()
                parts = []
            else:
                parts.append(line.upper())
    flush()

    if rec_id is None:
        raise ValueError(
            f"{fasta_file}: no '>' header found -- not a FASTA file"
        )
    if not records:
        raise ValueError(f"{fasta_file}: no records parsed")
    short = [(i, len(s)) for i, s in records if len(s) < min_example_len]
    if short:
        detail = ", ".join(f"{i} ({n} nt)" for i, n in short[:5])
        more = f" and {len(short) - 5} more" if len(short) > 5 else ""
        raise ValueError(
            f"{fasta_file}: {len(short)} record(s) shorter than "
            f"--min_example_len={min_example_len}: {detail}{more}. A short record "
            f"usually means a parsing problem; check the file or lower the threshold."
        )
    return records


def compose_icl_prompt(
    examples: list[tuple[str, str]],
    spacer: str,
    n_shots: int,
    rng: np.random.RandomState,
    spacer_both_sides: bool = False,
    fixed: bool = False,
) -> tuple[str, list[str], list[str]]:
    if n_shots > len(examples):
        n_shots = len(examples)
    if fixed:
        picked = examples[:n_shots]
    else:
        # Index into `examples`: rng.choice on a list of pairs would build a 2-D array.
        idx = rng.choice(len(examples), size=n_shots, replace=False)
        picked = [examples[int(i)] for i in idx]
    selected_ids = [i for i, _ in picked]
    selected = [s for _, s in picked]
    parts: list[str] = []
    for seq in selected:
        parts.append(spacer + seq + spacer if spacer_both_sides else spacer + seq)
    parts.append(spacer)
    return "".join(parts), selected, selected_ids


def collect_fastas(args, parser) -> list[tuple[str, str]]:
    if args.fasta:
        return [(os.path.splitext(os.path.basename(args.fasta))[0], args.fasta)]
    if args.fasta_dir:
        out = []
        for fname in sorted(os.listdir(args.fasta_dir)):
            if not fname.endswith(args.fasta_suffix):
                continue
            stem = fname.removesuffix(args.fasta_suffix)
            if args.cell_types and stem not in args.cell_types:
                continue
            out.append((stem, os.path.join(args.fasta_dir, fname)))
        return out
    parser.error("Provide either --fasta or --fasta_dir")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def build_llm(args, max_len: int):
    """The one validated engine configuration (vllm_evo2.api.engine_kwargs), plus this
    script's batch knob: vLLM keeps --batch_size requests resident and refills a slot
    as soon as one finishes, so distinct prompts run concurrently. Prefix caching
    (on) is vLLM's more general replicate_cached_data."""
    vllm_evo2.register()
    from vllm import LLM
    from vllm_evo2.api import engine_kwargs

    over = dict(max_num_seqs=args.batch_size)
    if args.max_num_batched_tokens:
        over["max_num_batched_tokens"] = args.max_num_batched_tokens
    kw = engine_kwargs(args.model, max_len, args.gpu_frac, args.kv_cache_dtype, **over)
    if args.no_prefix_caching:
        kw["enable_prefix_caching"] = False
    if args.eager:
        kw["enforce_eager"] = True
    return LLM(**kw)


def fmt(n) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batched ICL generation for Evo 2 under vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model
    parser.add_argument("--model", default="ckpt/evo2_7b_hf",
                        help="converted checkpoint DIRECTORY (not an evo2 model name)")

    # Data
    parser.add_argument("--fasta", default=None, help="Single FASTA file")
    parser.add_argument("--fasta_dir", default=None, help="Directory of FASTA files")
    parser.add_argument("--fasta_suffix", default=".fasta",
                        help="Suffix filter when using --fasta_dir")
    parser.add_argument("--cell_types", nargs="+", default=None,
                        help="Restrict to these cell types (stem of filename)")
    parser.add_argument("--min_example_len", type=int, default=1000,
                        help="Error out if any FASTA record is shorter than this. A "
                             "short record almost always means a parsing problem "
                             "(e.g. wrapped lines read as separate examples).")

    # Prompt composition
    parser.add_argument("--spacer", default="N",
                        help="Delimiter placed between examples (default: N)")
    parser.add_argument("--spacer_both_sides", action="store_true",
                        help="Wrap each example on both sides; the default is "
                             "left side only")
    parser.add_argument("--spacer_left_only", action="store_true",
                        help="Spacer only on the left side (the default)")
    parser.add_argument("--n_shots", type=int, nargs="+", default=[16],
                        help="Number of ICL examples (several = sweep)")
    parser.add_argument("--n_prompts", type=int, default=1,
                        help="Different random prompts per (fasta, n_shots)")
    parser.add_argument("--fixed_examples", action="store_true",
                        help="Use the first n_shots examples instead of sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for ICL example selection")

    # Generation
    parser.add_argument("--gen_length", type=int, default=230)
    parser.add_argument("--n_seqs", type=int, default=1,
                        help="Samples per composed prompt (default 1). Each sample is "
                             "its own vLLM request: SamplingParams(n>1) re-prefills "
                             "per sample instead of sharing the prompt.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="vLLM max_num_seqs: how many requests decode "
                             "concurrently; they need not share a prompt.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--gen_seed", type=int, default=None,
                        help="Seed sampling too. Off by default: a fixed seed on "
                             "identical prompts returns identical sequences, so this "
                             "derives a distinct per-request seed from the base value.")

    # Engine knobs
    parser.add_argument("--gpu_frac", type=float, default=0.85)
    parser.add_argument("--eager", action="store_true",
                        help="disable CUDA graphs (~1.4-1.65x slower decode)")
    parser.add_argument("--kv_cache_dtype", default="auto",
                        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                        help="fp8 doubles KV capacity and still passes the upstream gate")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                        help="prefill chunk size (vLLM default 16384)")
    parser.add_argument("--no_prefix_caching", action="store_true")

    # Output
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_fasta", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="compose prompts and print shapes; never touches the GPU")

    args = parser.parse_args()
    if args.spacer_left_only:
        args.spacer_both_sides = False

    fasta_files = collect_fastas(args, parser)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ---- compose every prompt up front -----------------------------------
    # One flat request list across all (fasta, n_shots, prompt, sample) cells is
    # the whole point: vLLM schedules them together.
    reqs: list[dict] = []
    skipped: dict[str, dict] = {}
    t0 = time.time()
    for label, fasta_path in fasta_files:
        examples = load_examples_from_fasta(fasta_path, args.min_example_len)
        print(f"[icl] {label}: {len(examples)} examples from {fasta_path}")
        for n_shots in args.n_shots:
            tag = f"{label}_shot{n_shots}"
            pkl_path = os.path.join(args.output_dir, f"{tag}.pkl")
            if os.path.exists(pkl_path):
                print(f"  {tag}: already exists, skipping")
                with open(pkl_path, "rb") as f:
                    skipped[tag] = pickle.load(f)
                continue
            for p_idx in range(args.n_prompts):
                prompt, selected, selected_ids = compose_icl_prompt(
                    examples, args.spacer, n_shots, rng,
                    args.spacer_both_sides, fixed=args.fixed_examples,
                )
                for s_idx in range(args.n_seqs):
                    reqs.append({
                        "tag": tag, "label": label, "n_shots": n_shots,
                        "fasta": fasta_path, "prompt_idx": p_idx,
                        "prompt": prompt, "selected": selected,
                        "selected_ids": selected_ids,
                    })
    t_compose = time.time() - t0

    if not reqs:
        print("[icl] nothing to do -- every output already exists")
        return

    lengths = [len(r["prompt"]) for r in reqs]
    n_tags = len({r["tag"] for r in reqs})
    print(f"\n[icl] {fmt(len(reqs))} requests over {n_tags} (fasta, n_shots) cells | "
          f"prompt tokens {min(lengths)}-{max(lengths)} "
          f"(total {fmt(sum(lengths))}) | gen_length={args.gen_length} | "
          f"batch_size={args.batch_size}")
    if args.dry_run:
        print("[icl] --dry_run: stopping before the engine loads")
        return

    # ---- one engine, one generate call -----------------------------------
    t0 = time.time()
    llm = build_llm(args, max(lengths) + args.gen_length + 16)
    t_load = time.time() - t0

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=list(r["prompt"].encode())) for r in reqs]
    base = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                max_tokens=args.gen_length)
    if args.gen_seed is None:
        params = SamplingParams(**base)
    else:
        # Per-request seeds: one shared seed over identical prompts would return
        # n_seqs copies of the same sequence.
        params = [SamplingParams(seed=args.gen_seed * 1_000_003 + i, **base)
                  for i in range(len(reqs))]

    t0 = time.time()
    outs = llm.generate(prompts, params)
    t_gen = time.time() - t0

    # ---- regroup and write ------------------------------------------------
    results: dict[str, dict] = dict(skipped)
    gen_tokens = 0
    cached_tokens = 0
    for r, o in zip(reqs, outs):
        c = o.outputs[0]
        gen_tokens += len(c.token_ids)
        cached_tokens += getattr(o, "num_cached_tokens", 0)
        res = results.setdefault(r["tag"], {
            "label": r["label"], "n_shots": r["n_shots"], "sequences": [],
            "prompt_examples": [], "prompt_example_ids": [],
            "prompt_idx": [], "n_prompts": args.n_prompts,
            "examples_file": r["fasta"], "checkpoint": None, "model": args.model,
        })
        res["sequences"].append(bytes(c.token_ids).decode("ascii", "replace"))
        res["prompt_examples"].append(r["selected"])
        res["prompt_example_ids"].append(r["selected_ids"])
        res["prompt_idx"].append(r["prompt_idx"])

    for tag in {r["tag"] for r in reqs}:
        res = results[tag]
        with open(os.path.join(args.output_dir, f"{tag}.pkl"), "wb") as f:
            pickle.dump(res, f)
        if args.save_fasta:
            with open(os.path.join(args.output_dir, f"{tag}.fasta"), "w") as f:
                for i, s in enumerate(res["sequences"]):
                    f.write(f">{tag}_{i}\n{s}\n")
    with open(os.path.join(args.output_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # ---- timing -----------------------------------------------------------
    prompt_tokens = sum(lengths)
    print(f"\n=== speed ({len(reqs)} requests, batch_size={args.batch_size}) ===")
    print(f"  compose prompts     : {t_compose:7.2f} s")
    print(f"  engine init         : {t_load:7.2f} s   (weights + CUDA graphs, paid once)")
    print(f"  generate            : {t_gen:7.2f} s")
    print(f"  prompt tokens       : {fmt(prompt_tokens):>12}  "
          f"({fmt(cached_tokens)} served from the prefix cache, "
          f"{100 * cached_tokens / max(prompt_tokens, 1):.1f}%)")
    print(f"  generated tokens    : {fmt(gen_tokens):>12}")
    print(f"  throughput          : {gen_tokens / t_gen:8.0f} gen tok/s   "
          f"{(prompt_tokens + gen_tokens) / t_gen:8.0f} tok/s incl. prompt")
    print(f"                        {len(reqs) / t_gen:8.2f} seq/s   "
          f"{t_gen / len(reqs):8.3f} s/seq")

    print("\n=== output ===")
    for tag in sorted(results):
        seqs = results[tag]["sequences"]
        valid = sum(1 for s in seqs if s and set(s.upper()) <= set("ACGT"))
        gc = [(s.upper().count("G") + s.upper().count("C")) / len(s) for s in seqs if s]
        print(f"  {tag:<28} {len(seqs):>5} seqs  {valid}/{len(seqs)} valid DNA  "
              f"mean GC={np.mean(gc):.3f}")
    print(f"\n[icl] wrote {os.path.join(args.output_dir, 'all_results.pkl')}")


if __name__ == "__main__":
    main()
