# Scoring oracles

Each ICL task is scored by a published activity model. These are other
people's models, under their own licences, and none of them is redistributed
in this repository — obtain each from its publication, place it under this
directory in the layout below, then `source oracles/env.sh`.

Everything here except this file and `env.sh.example` is gitignored, so
weights placed in this directory cannot be committed by accident.

| Task | Oracle | Predicts |
|---|---|---|
| yeast promoter | regLM | yeast promoter activity (Enformer regression head) |
| human 5'UTR | GEMORNA | translational capacity / mean ribosome load |
| human promoter | PARM | promoter activity, per cell type, 5-fold ensemble |
| in-vitro enhancer | Malinois | MPRA enhancer activity, per cell type |

## Where to get them

**regLM** — Lal, A., Garfield, D., Biancalani, T., & Eraslan, G. (2024).
*Designing realistic regulatory DNA with autoregressive language models.*
Genome Research. Released by Genentech under the MIT licence.
This project uses the yeast-promoter regression checkpoint
(`yeast_regression_separate_defined.ckpt`), not the generative model.

**GEMORNA** — Zhang, H., Liu, H., Xu, Y., Huang, H., Liu, Y., Wang, J.,
Qin, Y., Wang, H., Ma, L., Xun, Z., Hou, X., Lu, T. K., & Cao, J. (2025).
*Deep generative models design mRNA sequences with enhanced translational
capacity and stability.* Science.
Licensed for non-commercial research use only. Request access from the
authors; it may not be redistributed.

**PARM** — Barbadilla-Martínez, L., et al. (2026). *Regulatory grammar in
human promoters uncovered by MPRA-based deep learning.* Nature 651(8107),
1107–1116. https://doi.org/10.1038/s41586-025-10093-z
(The 2024 bioRxiv preprint carried the earlier title "The regulatory grammar
of human promoters uncovered by MPRA-trained deep learning".)
Distributed under an evaluation licence from Oncode/UMCU; it may not be
redistributed.

**Malinois** (CODA) — Gosai, S. J., Castro, R. I., Fuentes, N., Butts, J. C.,
Mouri, K., Alasoadura, M., Kales, S., Nguyen, T. T. L., Noche, R. R.,
Rao, A. S., Joy, M. T., Sabeti, P. C., Reilly, S. K., & Tewhey, R. (2024).
*Machine-guided design of cell-type-targeting cis-regulatory elements.*
Nature 634(8036), 1211–1220. https://doi.org/10.1038/s41586-024-08070-z
The model weights are released under the MIT licence; the surrounding
toolkit is AGPL-3.0.

## Redistribution

GEMORNA and PARM may **not** be redistributed. Keep them out of the git
history and out of any release asset or model-hub upload. A consequence worth
stating plainly: the 5'UTR and human-promoter scoring stages cannot be
reproduced by anyone who has not separately obtained those two models.

regLM and the Malinois weights are MIT and could be redistributed with their
copyright notices, but are kept out of this repository for consistency.

Generation, novelty metrics, and GRPO with the dependency-free rewards need
none of these.

## Expected layout

```
oracles/
  reglm/                                $REGLM_REPO (placed on sys.path)
    scripts/regression.py               imported as scripts.regression
    yeast_promoters/03_regression_separate/
      yeast_regression_separate_defined.ckpt      $REGLM_YEAST_CKPT
  gemorna/
    checkpoints/5utr.pt                 $GEMORNA_5UTR_CKPT
  parm/
    pre_trained_models/<CELL>/          $PARM_MODEL_DIR is the parent dir;
                                        one 5-fold ensemble per cell type
  malinois/                             $MALINOIS_DIR
    MODELS-malinois_artifacts__*.tar.gz $MALINOIS_ARTIFACT
```

`regLM/scripts/regression.py` needs `enformer-pytorch` and
`pytorch-lightning`, and has no intra-repo imports — only `scripts/` and the
checkpoint are required, not a full clone.

PARM covers AGS, HAP1, HCT116, HEK293, HepG2, K562, LNCaP, MCF7 and U2OS.
Each cell-type directory holds five `.parm` files, one per fold; do not
rename them.

## Use

```bash
cp oracles/env.sh.example oracles/env.sh    # edit if your layout differs
source oracles/env.sh
```

Any stage that needs an oracle it cannot find exits immediately, naming the
environment variable that is unset. Stages that do not score are unaffected.
