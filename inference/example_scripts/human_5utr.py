#!/usr/bin/env python
"""Human 5'UTR design by in-context learning.

Scored with GEMORNA for translational capacity.

    python example_scripts/human_5utr.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_scripts.pipeline import run

run(dict(
    name="human_5utr",
    description="human 5'UTRs, scored with GEMORNA",

    fasta="data/5UTR/5UTR.fasta",
    fasta_dir=None,
    prompt_mode="top_n",

    ckpt="evo2_7b_131kb_gen_hf",
    spacer="ATGCATGA",
    spacer_left_only=False,
    n_shots=[4, 8, 16, 32, 64, 128],
    gen_length=50,
    n_seqs=512,
    batch_size=256,
    temperature=1.0,
    gpu_frac=0.85,

    gc_min=None, gc_max=0.65,
    seq_id_max=0.6, blastn_max=30,
    homopolymer_max_run=8, no_spacer_boundary=16,

    scorer="gemorna",
    selected_threshold=6.0,
))
