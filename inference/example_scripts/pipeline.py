"""The four stages every example task runs, end to end.

    FASTA pool -> prompt sets -> Evo 2 generation -> filter + score -> table

Why this orchestrates subprocesses instead of just calling functions: the
stages need **incompatible environments**.  Generation wants torch 2.11 with
vLLM; some scoring oracles want torch 2.7.  They cannot
coexist in one interpreter, so each stage is launched in its own conda
environment, named by `$ENV_GEN` and `$ENV_METRICS`.

Run a whole task:

    python example_scripts/human_5utr.py

or one stage at a time, from the right environment yourself:

    python example_scripts/human_5utr.py --stage prompt      # any env
    python example_scripts/human_5utr.py --stage generate    # $ENV_GEN
    python example_scripts/human_5utr.py --stage score       # $ENV_METRICS
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE = os.path.dirname(HERE)
REPO = os.path.dirname(INFERENCE)
sys.path.insert(0, INFERENCE)


# ---------------------------------------------------------------- helpers --
def _conda_run(env_var, argv, cwd=INFERENCE):
    """Run argv inside the conda env named by $<env_var>.

    env_var None, or the variable unset, runs in place: the caller is assumed
    to have activated the right environment already, which is what --stage is
    for.
    """
    env_name = os.environ.get(env_var) if env_var else None
    if env_name:
        cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + argv
    else:
        if env_var:
            print(f"[pipeline] ${env_var} unset; running in the current "
                  f"environment and assuming it is correct")
        cmd = argv
    print(f"[pipeline] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"stage failed ({r.returncode}): {' '.join(cmd)}")


def ckpt_path(name):
    root = os.environ.get("GENOMIC_ICL_CKPT", os.path.join(REPO, "ckpt"))
    p = os.path.join(root, name)
    if not os.path.isdir(p):
        raise SystemExit(
            f"checkpoint {p} not found. See {REPO}/ckpt/README.md for how to "
            f"build it, or set $GENOMIC_ICL_CKPT.")
    return p


# ------------------------------------------------------------- the stages --
def stage_prompt(task):
    """FASTA pool(s) -> one FASTA per example set."""
    argv = [sys.executable, "prompt.py", "--mode", task["prompt_mode"],
            "--set_size", str(max(task["n_shots"])),
            "--out", task["prompt_dir"], "--overwrite"]
    if task["fasta"]:
        argv += ["--fasta", task["fasta"]]
    else:
        argv += ["--fasta_dir", task["fasta_dir"]]
    if task["prompt_mode"] != "top_n":
        argv += ["--n_sets", str(task.get("n_sets", 32)),
                 "--candidate_top", str(task.get("candidate_top", 300))]
    _conda_run(None, argv)             # stdlib only; no env needed


def stage_generate(task):
    """Prompt sets -> sampled sequences, one pkl per (set, shot count)."""
    argv = [sys.executable, "Evo2_ICL.py",
            "--model", ckpt_path(task["ckpt"]),
            "--fasta_dir", task["prompt_dir"], "--fasta_suffix", ".fasta",
            "--fixed_examples",
            "--spacer", task["spacer"],
            "--n_shots", *[str(k) for k in task["n_shots"]],
            "--gen_length", str(task["gen_length"]),
            "--n_seqs", str(task["n_seqs"]),
            "--batch_size", str(task["batch_size"]),
            "--temperature", str(task["temperature"]),
            "--gpu_frac", str(task["gpu_frac"]),
            "--output_dir", task["gen_dir"]]
    if not task.get("spacer_left_only"):
        argv += ["--spacer_both_sides"]
    _conda_run("ENV_GEN", argv)


def stage_score(task):
    """Generated sequences -> filtered, scored table."""
    argv = [sys.executable, "-m", "example_scripts.score",
            "--gen_dir", task["gen_dir"],
            "--out", task["out_csv"],
            "--task_json", task["_json"]]
    _conda_run("ENV_METRICS", argv)


STAGES = {"prompt": stage_prompt, "generate": stage_generate,
          "score": stage_score}


def run(task):
    """Entry point for an example script: parse --stage and dispatch."""
    ap = argparse.ArgumentParser(description=task["description"])
    ap.add_argument("--stage", choices=list(STAGES),
                    help="run one stage only (default: all three, in order)")
    ap.add_argument("--out_root", default=None,
                    help="where to write prompts/, gen/ and the final table")
    args = ap.parse_args()

    root = args.out_root or os.environ.get(
        "GENOMIC_ICL_OUT", os.path.join(INFERENCE, "out"))
    base = os.path.join(root, task["name"])
    task["prompt_dir"] = os.path.join(base, "prompts")
    task["gen_dir"] = os.path.join(base, "gen")
    task["out_csv"] = os.path.join(base, f"{task['name']}_scored.csv")
    os.makedirs(base, exist_ok=True)

    import json
    task["_json"] = os.path.join(base, "task.json")
    with open(task["_json"], "w") as f:
        json.dump({k: v for k, v in task.items() if not k.startswith("_")},
                  f, indent=2)

    print(f"=== {task['name']}: {task['description']}")
    print(f"    output -> {base}")
    for name in ([args.stage] if args.stage else list(STAGES)):
        print(f"\n--- stage: {name}")
        STAGES[name](task)
    print(f"\n[done] {task['out_csv']}")
