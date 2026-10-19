# Genomic ICL

In-context learning with [Evo 2](https://github.com/arcinstitute/evo2) for
genomic sequence design, and reinforcement-learning post-training to improve
it. Two halves:

| | |
|---|---|
| [`inference/`](inference/) | Tooling for in-context learning with Evo 2 on the vLLM plugin — prompt construction, batched generation, filtering and activity scoring — plus four worked examples (yeast promoter, human 5'UTR, human promoter, in-vitro MPRA enhancer). |
| [`post_training/`](post_training/) | GRPO and DPO post-training of Evo 2 for ICL design, with a pluggable reward system (activity oracles, GC content, sequence identity, k-mer entropy). |
| [`ckpt/`](ckpt/) | Converted Evo 2 checkpoints. Not in git — see [`ckpt/README.md`](ckpt/README.md). |
| [`oracles/`](oracles/) | The four published activity models used for scoring. Not in git — see [`oracles/README.md`](oracles/README.md). |

## Upstream dependencies

- **Evo 2** — https://github.com/arcinstitute/evo2 (the `evo2` package and its
  `vortex` backend; not on PyPI)
- **Evo 2 vLLM plugin** — (used via
  `sys.path`, not pip-installed)

## Two environments, not one

The halves need incompatible stacks and cannot share an environment:

| | `inference/` | `post_training/` |
|---|---|---|
| torch | 2.11 + cu129 | 2.7 |
| engine | vLLM 0.26 + the evo2-vllm plugin | vortex + TransformerEngine |
| installs as | scripts, run in place | `pip install -e post_training/` |

The same split appears inside `inference/`: generation needs vLLM, scoring
needs the oracle's own dependencies. `inference/README.md` defines two roles —
`ENV_GEN` and `ENV_METRICS` — and says what each must contain; you name your
own environments through those variables.

## Getting started

```bash
export GENOMIC_ICL=$PWD
export EVO2_VLLM_REPO=/path/to/evo2-vllm

# Run an in-context-learning example end to end.
python inference/example_scripts/human_5utr.py

# Post-training.
pip install -e post_training/ && source post_training/setup_env.sh
python -m dna_rl.cli.train --config post_training/configs/grpo/evo2/human_promoter_KLD_Nspacer.yaml
```

Each half's README covers its own layout, environment variables and commands.

## What is not in this repository

- **Model weights.** ~27 GB of converted Evo 2 checkpoints; rebuild them with
  the converter or fetch the release assets. See [`ckpt/README.md`](ckpt/README.md).
- **Scoring oracles.** regLM, GEMORNA, PARM and Malinois are published
  third-party models under their own licences;
  [`oracles/README.md`](oracles/README.md) gives the paper to cite and
  download each from. GEMORNA and PARM may not be redistributed, so the 5'UTR
  and human-promoter scoring stages need models you obtain yourself. Every
  oracle is reached through an environment variable and the code raises an
  error naming that variable the first time it is needed, so generation,
  novelty metrics and the dependency-free rewards all run on a bare clone.
  The EvoEnhancer and AlphaGenome reward backends work the same way.

## License

See [`LICENSE`](LICENSE) — **not yet chosen**; this must be settled before the
repository is made public.
