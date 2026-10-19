# Checkpoints

Evo 2 weights converted to the HF/safetensors layout the
[evo2-vllm] plugin loads. The
directories themselves are **not in git** — rebuild them with the commands
below, or fetch them from the release assets.

`inference/replication_config.py` finds this directory via `$GENOMIC_ICL_CKPT`,
defaulting to `<repo>/ckpt`.

| Directory | Size | Source | Used by |
|---|---|---|---|
| `evo2_7b_base_hf` | 13 G | `arcinstitute/evo2_7b_base` | the yeast-promoter task |
| `evo2_7b_131kb_gen_hf` | 13 G | Evo 2 7B 131kb-gen | 5'UTR, human promoter, enhancer |
| `evo2_1b_base_fp8_hf` | 1.9 G | `arcinstitute/evo2_1b_base` | every 1B arm |

## Rebuilding

```bash
export EVO2_VLLM_REPO=/path/to/evo2-vllm     
cd "$EVO2_VLLM_REPO"

# Downloads from HuggingFace on first run (~13 GB, ~10 min on one H100).
python scripts/convert_evo2_to_hf.py --model evo2_7b_base \
    --out "$GENOMIC_ICL_CKPT/evo2_7b_base_hf"
```

## The 1B must be the FP8 conversion

Evo 2's 1B was trained with FP8 Hyena input projections. The plain bf16
conversion of it (`evo2_1b_base_hf`) is numerically dead — it scores a
perplexity of 4.02, essentially the ln(4) ≈ 1.386 floor for a 4-letter
alphabet, i.e. no signal at all. **Only `evo2_1b_base_fp8_hf` is shipped**, and
only `main`-branch evo2-vllm can produce it; the conversion exports selective
static E4M3 W8A8 weights using the checkpoint's own TransformerEngine amax.

One consequence: `inference/compare_models.py` (stage 6b) and
`inference/compare_separators.py` (stage 6c) were run against the bf16
checkpoint and **cannot be reproduced from this release**. Their findings were
retracted and superseded by `compare_modalities.py` (stages 6d/6e) on the FP8
checkpoint. The code is kept because it is the record of how that was caught.

The 7B is the opposite case: its GRPO runs were multi-GPU, which turns FP8 off
during training, so 7B fine-tunes convert in bf16.

## Scoring oracles

The activity models that score generated sequences are separate third-party
models and live under `../oracles/`, not here. See
[`../oracles/README.md`](../oracles/README.md) for the publication to cite and
download each one from, and for which of them may be redistributed.
