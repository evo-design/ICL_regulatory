# In-context learning with Evo 2

Evo 2 does in-context learning on DNA: show it a train of example sequences,
delimited by a spacer, and it continues the pattern. This directory is the
tooling for doing that at scale on the
[Evo 2 vLLM plugin], plus four worked
examples you can run end to end and adapt.

```
FASTA pool  ->  prompt sets  ->  generation  ->  filter + score  ->  CSV
             prompt.py        Evo2_ICL.py      filters.py / oracles.py
```

## The pieces

| | |
|---|---|
| `prompt.py` | Choose which examples go in each prompt. Three modes — `top_n`, `nested`, `random`. Writes one FASTA per set. |
| `Evo2_ICL.py` | Compose prompts from those sets and sample with vLLM. Vendored from the plugin's `generate_icl_vllm.py`. |
| `filters.py` | Sequence metrics (GC, homopolymer runs, spacer leakage) and novelty against the prompt's own examples. |
| `oracles.py` | Score designs with one of four published activity models. |
| `scorers/` | The oracle implementations. |
| `paths.py` | Where the oracle weights live (all from environment variables). |
| `data/` | The example pools used by the four worked examples. |
| `example_scripts/` | The four end-to-end examples. |

## Quickstart

```bash
export EVO2_VLLM_REPO=/path/to/evo2-vllm      # the plugin
export GENOMIC_ICL_CKPT=/path/to/ckpt         # default: ../ckpt
source ../oracles/env.sh                      # only needed for scoring

python example_scripts/human_5utr.py
```

That runs all three stages. Because generation and scoring need
**incompatible environments**, each stage is launched in its own conda
environment, named by `$ENV_GEN` and `$ENV_METRICS`. Set those and the script
handles the switching; leave them unset and each stage runs in your current
environment, which is what you want when driving stages by hand:

```bash
python example_scripts/human_5utr.py --stage prompt      # any environment
python example_scripts/human_5utr.py --stage generate    # needs vLLM
python example_scripts/human_5utr.py --stage score       # needs the oracle
```

| Variable | Environment that must contain |
|---|---|
| `ENV_GEN` | torch 2.11, vLLM 0.26, and the evo2-vllm plugin importable |
| `ENV_METRICS` | biopython, pandas, and oracles |

## The four examples

| Script | Designs | Pool | Prompts | Oracle |
|---|---|---|---|---|
| `yeast_promoter.py` | yeast promoters, 80 bp | 300 yeast promoters | `top_n` | regLM |
| `human_5utr.py` | human 5'UTRs, 50 bp | 200 5utrs | `top_n` | GEMORNA |
| `human_promoter.py` | human promoters, 230 bp | 9 cell lines, activity-ranked | `top_n` | PARM |
| `invitro_enhancer.py` | MPRA enhancers, 200 bp | 3 cell lines | `top_n` | Malinois |

Each script is a single dict of parameters — pool, spacer, shot counts,
sampling, filter thresholds, oracle — followed by one `run(...)` call. To
adapt one to your own data, copy it and change the pool and the thresholds.

The oracles are third-party models under their own licences and are not
redistributed here; see [`../oracles/README.md`](../oracles/README.md). The
prompt and generation stages need none of them.

## Compiling prompts

`prompt.py` compiles ICL prompts.

The draw is a hash-ordered permutation rather than an RNG call — candidates
sorted by `sha256(seed | label | set | index)` — so it reproduces bit for bit
on any machine, and `prompt_sets.json` records the pool indices behind every
set.

## Using the pieces directly

Nothing requires the example scripts.

```bash
# 32 nested sets of 128 examples from your own pool
python prompt.py --fasta my_pool.fasta --mode nested \
    --n_sets 32 --candidate_top 300 --set_size 128 --out out/prompts

# sample from them; the spacer defaults to a single N on the left of each
# example, so pass --spacer / --spacer_both_sides only to change that
python Evo2_ICL.py --model $GENOMIC_ICL_CKPT/evo2_7b_131kb_gen_hf \
    --fasta_dir out/prompts --fasta_suffix .fasta --fixed_examples \
    --n_shots 4 8 16 32 --gen_length 200 \
    --n_seqs 256 --output_dir out/gen

# filter and score
python -m example_scripts.score --gen_dir out/gen \
    --out out/scored.csv --task_json out/task.json
```

`Evo2_ICL.py --save_fasta` also writes plain FASTA if you would rather filter
with your own tooling. `python paths.py` prints which oracle weights it found.

