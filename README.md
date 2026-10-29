# In-context learning for regulatory sequence design

In-context learning with [Evo 2](https://github.com/arcinstitute/evo2) for
regulatory sequence design, and reinforcement-learning post-training to improve
it. The repo contains two parts:

| | |
|---|---|
| [`inference/`](inference/) |  In-context learning with Evo 2 and the vLLM engine — prompt construction, batched generation, filtering and activity scoring — plus four worked examples (yeast promoter, human 5'UTR, human promoter, in-vitro MPRA enhancer). |
| [`post_training/`](post_training/) | GRPO post-training of Evo 2 for ICL design, with a custimized reward system (activity oracles, GC content, sequence identity). |
| [`ckpt/`](ckpt/) | Converted Evo 2 checkpoints. See [`ckpt/README.md`](ckpt/README.md). |
| [`oracles/`](oracles/) | The four published activity models used for scoring. See [`oracles/README.md`](oracles/README.md). |

## Upstream dependencies

- **Evo 2** — https://github.com/arcinstitute/evo2 (the `evo2` package and its
  `vortex` backend)
- **Evo 2 vLLM engine** — https://github.com/evo-design/evo2_flash

## Environments

The two parts need two stacks and may not share an environment:

| | `inference/` | `post_training/` |
|---|---|---|
| torch | 2.11 + cu129 | 2.7 |
| engine | vLLM 0.26 + the evo2-vllm engine | vortex + TransformerEngine |
| installs as | scripts, run in place | `pip install -e post_training/` |

The same split appears inside `inference/`: generation needs vLLM, scoring
needs the oracle's own dependencies. `inference/README.md` defines two environments —
`ENV_GEN` and `ENV_METRICS` and says what each must contain. You can name your
own environments through those variables.

## Getting started

```bash
export GENOMIC_ICL=$PWD
export EVO2_VLLM_REPO=/path/to/evo2-vllm

# Run an in-context-learning example end to end.
python inference/example_scripts/human_5utr.py

# Post-training.
pip install -e post_training/ && source post_training/setup_env.sh
python -m dna_rl.cli.train --config post_training/configs/grpo/evo2/human_promoter.yaml
```

Each half's README covers its own layout, environment variables and commands.

## Not in this repository

- **Model weights.** ~27 GB of converted Evo 2 checkpoints; rebuild them with
  the converter or fetch the release assets. See [`ckpt/README.md`](ckpt/README.md).
- **Scoring oracles.** regLM, GEMORNA, PARM and Malinois are published
  third-party models under their own licences.
  [`oracles/README.md`](oracles/README.md) gives the paper to cite and
  download each from. GEMORNA and PARM may not be redistributed, so the 5'UTR
  and human-promoter scoring stages need models you obtain yourself. Every
  oracle is reached through an environment variable and the code raises an
  error naming that variable the first time it is needed, so generation,
  novelty metrics and the dependency-free rewards all run on a bare clone.
  Please cite research articles for these oracles properly if you use them.

## License

MIT license.
