#!/usr/bin/env python
"""Yeast promoter design by in-context learning.

Shows Evo 2 a run of yeast promoters and asks it to continue the pattern;
designs are then filtered and scored with the regLM activity model.

    python example_scripts/yeast_promoter.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_scripts.pipeline import run

run(dict(
    name="yeast_promoter",
    description="yeast promoters, scored with regLM",

    # --- the example pool -------------------------------------------------
    fasta="data/yeast_promoter/yeast_random.fasta",
    fasta_dir=None,
    prompt_mode="top_n",          

    # --- generation -------------------------------------------------------
    ckpt="evo2_7b_base_hf",
    spacer="ATGCATGACA",         
    spacer_left_only=False,
    n_shots=[1, 2, 4, 8, 16, 32, 64],
    gen_length=80,
    n_seqs=1024,                  # samples per (prompt, shot count)
    batch_size=256,
    temperature=0.8,
    gpu_frac=0.85,

    # --- filtering --------------------------------------------------------
    gc_min=None, gc_max=0.65,
    seq_id_max=0.6, blastn_max=50,
    homopolymer_max_run=8, no_spacer_boundary=20,

    # --- scoring ----------------------------------------------------------
    scorer="reglm",
    selected_threshold=12.0,
))
