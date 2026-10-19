#!/usr/bin/env python
"""In-vitro enhancer design by in-context learning, per cell type.

Scored with Malinois (MPRA activity) for HepG2, K562 and SK-N-SH.

    python example_scripts/invitro_enhancer.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_scripts.pipeline import run

run(dict(
    name="invitro_enhancer",
    description="in-vitro MPRA enhancers in 3 cell lines, scored with Malinois",

    fasta=None,
    fasta_dir="data/invitro_cell_activity_seqprop",   
    prompt_mode="top_n",

    ckpt="evo2_7b_131kb_gen_hf",
    spacer="ATGCATGA",
    spacer_left_only=False,
    n_shots=[2, 4, 8, 16, 32, 64],
    gen_length=200,
    n_seqs=512,
    batch_size=128,
    temperature=1.0,
    gpu_frac=0.85,

    gc_min=0.25, gc_max=0.65,
    seq_id_max=0.6, blastn_max=60,
    homopolymer_max_run=8, no_spacer_boundary=20,

    scorer="malinois",
    selected_threshold=3.0,
))
