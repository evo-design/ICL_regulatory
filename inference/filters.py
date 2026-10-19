import numpy as np
import pandas as pd


# ---------------------------------------------------------------- metrics --
def get_GC_content(seq):
    return (seq.count('C') + seq.count('G')) / len(seq)


def get_reverse_complement(seq):
    return seq[::-1].translate(str.maketrans('ATCG', 'TAGC'))


def valid_sequence(seq):
    return seq.count('A') + seq.count('T') + seq.count('C') + seq.count('G') == len(seq)


def homopolymer_ok(s, max_run=8):
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
    return True


def no_spacer(s, boundary_len, spacer_seq):
    """True iff the spacer motif is absent from the sequence interior.

    boundary_len is per-task (notebooks use 40 for human_promoter, 20 for the
    enhancer) and is NOT derived from the spacer length.
    """
    if boundary_len is None:
        boundary_len = 2 * len(spacer_seq)
    return spacer_seq not in s[boundary_len:-boundary_len]


def get_local_alignment_scores(seq_A, lib_B, rc=True):
    """Max-per-example local blastn score, taking the better of fwd / revcomp."""
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner("blastn")
    aligner.mode = 'local'
    scores = []
    for seq_B in lib_B:
        fwd = next(aligner.align(seq_A, seq_B)).score
        if rc:
            rev = next(aligner.align(seq_A, get_reverse_complement(seq_B))).score
            fwd = max(fwd, rev)
        scores.append(fwd)
    return scores


def run_pairwise_blastn(seq_A, lib_B, mode='global'):    
    from Bio.Align import PairwiseAligner
    if mode not in ['global', 'local']:
        raise ValueError(f"Invalid mode: {mode}")

    aligner = PairwiseAligner("blastn")
    aligner.mode = mode
        
    seq_ids = []
    scores = []
    for seq_B in lib_B:
        # Align
        alignment = next(aligner.align(seq_A, seq_B))
        # Calculate Identity
        # Identity = Matches / Alignment Length
        matches = alignment.counts().identities
        length = alignment.length
        seq_ids.append(matches / length)
        scores.append(alignment.score)
    return seq_ids, scores


def novelty(seq, example_pool):
    """(max global %identity, max local blastn score incl. RC) vs the pool.

    `example_pool` must already be sliced to the examples that were actually in
    the prompt, i.e. pool[:n_shots].
    """
    seq_ids, _ = run_pairwise_blastn(seq, example_pool)      # global, fraction
    seq_id = float(np.max(seq_ids)) if len(seq_ids) else 0.0
    blastn = float(np.max(get_local_alignment_scores(seq, example_pool, rc=True)))
    return seq_id, blastn


def _novelty_worker(args):
    """Module-level so multiprocessing.Pool can pickle it."""
    seq, pool = args
    return novelty(seq, pool)


def novelty_many(pairs, n_jobs=8, chunksize=32):
    """pairs: iterable of (seq, sliced_pool).  Returns [(seq_id, blastn), ...]."""
    pairs = list(pairs)
    if n_jobs <= 1:
        return [_novelty_worker(p) for p in pairs]
    from multiprocessing import Pool
    with Pool(n_jobs) as p:
        return p.map(_novelty_worker, pairs, chunksize=chunksize)


def load_examples_from_fasta(fasta_file):
    """Header-skipping reader; one sequence per non-'>' line.

    Matches both ICL_sweep.py and generate_icl_vllm.py.  The pools are all
    single-line-per-record.
    """
    out = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                continue
            s = line.strip().upper()
            if s:
                out.append(s)
    return out


# ---------------------------------------------------------------- filters --
def add_filter_columns(df, tcfg):
    """Add the two sequence-shape columns the keep mask needs.

    `homopolymer_ok`  no run of a single base longer than homopolymer_max_run
    `no_spacer`       the delimiter does not appear inside the design, away
                      from its edges.  If it does, the "design" is really two
                      fragments the model glued together.
    """
    df = df.copy()
    df["homopolymer_ok"] = df["seq"].apply(
        lambda x: homopolymer_ok(x, tcfg["homopolymer_max_run"]))
    df["no_spacer"] = df["seq"].apply(
        lambda x: no_spacer(x, tcfg["no_spacer_boundary"], tcfg["spacer"]))
    return df


def keep_mask(df, tcfg):
    """Which generated sequences survive filtering.  All five criteria apply.

    Needs GC_ratio, sequence_identity and blastn_score on the frame, plus the
    two columns add_filter_columns writes.
    """
    m = pd.Series(True, index=df.index)
    if tcfg["gc_min"] is not None:
        m &= df["GC_ratio"] > tcfg["gc_min"]
    m &= df["GC_ratio"] < tcfg["gc_max"]
    m &= df["sequence_identity"] < tcfg["seq_id_max"]
    m &= df["blastn_score"] < tcfg["blastn_max"]
    m &= df["homopolymer_ok"]
    m &= df["no_spacer"]
    return m
