#!/usr/bin/env python
"""Human promoter design by in-context learning, per cell type.

One pool per cell line, each ranked by measured activity, so `top_n` takes
the k most active promoters as the prompt.  Scored with PARM, which has a
separate 5-fold ensemble per cell type.

    python example_scripts/human_promoter.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_scripts.pipeline import run

run(dict(
    name="human_promoter",
    description="human promoters in 9 cell lines, scored with PARM",

    fasta=None,
    fasta_dir="data/human_promoter",     # AGS.fasta, HAP1.fasta, ...
    prompt_mode="top_n",

    ckpt="evo2_7b_131kb_gen_hf",
    spacer="ATGCATGA",
    spacer_left_only=False,
    n_shots=[2, 4, 8, 16, 64, 128],
    gen_length=230,
    n_seqs=256,
    batch_size=128,
    temperature=1.0,
    gpu_frac=0.9,

    gc_min=0.25, gc_max=0.75,
    seq_id_max=0.6, blastn_max=60,
    homopolymer_max_run=8, no_spacer_boundary=40,

    scorer="parm",
    selected_threshold=4.0,
))
