"""The four in-silico oracles, one adapter each, aligned to the caller's row order.

Heavy dependencies are imported lazily inside each adapter so that a process
only ever loads the oracle its environment actually has:

    reglm     regLM EnformerModel, yeast promoter activity
    gemorna   GEMORNA 5'UTR MRL
    parm      PARM promoter activity, per cell type, 5-fold
    malinois  Malinois MPRA, per cell type, fwd+revcomp avg

Contract: score(task, df, device) -> np.ndarray of len(df), in df's order, no
NaNs.  `df` needs a `seq` column, plus `cell_type` for parm/malinois.

Score everything you intend to compare in one call per (scorer, cell_type):
a model reload or a batch boundary between two calls can shift scores enough
to look like a real difference.
"""

import os
import sys

import numpy as np

import paths

if paths.SCORERS_PARENT not in sys.path:
    sys.path.insert(0, paths.SCORERS_PARENT)


def _align_by_sequence(seqs, out_df, seq_col, val_col):
    """Map an oracle's output frame back onto the input order.
    """
    seqs = list(seqs)
    if len(out_df) == len(seqs) and list(out_df[seq_col]) == seqs:
        out = out_df[val_col].to_numpy(dtype=np.float64)
    else:
        lut = dict(zip(out_df[seq_col].tolist(), out_df[val_col].tolist()))
        out = np.array([lut.get(s, np.nan) for s in seqs], dtype=np.float64)
    if np.isnan(out).any():
        n = int(np.isnan(out).sum())
        raise RuntimeError(
            f"oracle returned no value for {n}/{len(seqs)} sequences; "
            f"pre-filter with valid_sequence() before scoring")
    return out


# ------------------------------------------------------------------ yeast --
def _score_reglm(df, device="cuda:0"):
    """regLM Enformer regression, 'defined media' checkpoint.
    """
    paths.require_path(paths.REGLM_REPO, "REGLM_REPO", "score yeast promoters")
    paths.require_path(paths.REGLM_YEAST_CKPT, "REGLM_YEAST_CKPT",
                     "score yeast promoters")
    if paths.REGLM_REPO not in sys.path:
        sys.path.append(paths.REGLM_REPO)
    import scripts.regression  # noqa: E402

    model = scripts.regression.EnformerModel.load_from_checkpoint(
        paths.REGLM_YEAST_CKPT, map_location=device)
    ds = scripts.regression.SeqDataset(df["seq"].tolist())
    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    preds = model.predict_on_dataset(ds, device=dev_idx, num_workers=8)
    out = np.asarray(preds).astype(np.float64).flatten()
    if out.shape[0] != len(df):
        raise RuntimeError(f"regLM returned {out.shape[0]} for {len(df)} rows")
    return out


# ------------------------------------------------------------------- 5utr --
def _score_gemorna(df, device="cuda:0"):
    paths.require_path(paths.GEMORNA_5UTR_CKPT, "GEMORNA_5UTR_CKPT",
                     "score 5'UTRs")
    from scorers.gemorna_5utr_score import GEMORNA5UTRScorer
    scorer = GEMORNA5UTRScorer(paths.GEMORNA_5UTR_CKPT, device=device)
    seqs = df["seq"].tolist()
    out = scorer.score_sequences(seqs, batch_size=512)
    return _align_by_sequence(seqs, out, "sequence", "scaled_score")


# --------------------------------------------------------- human promoter --
def _score_parm(df, device="cuda:0"):
    """PARM, one 5-fold ensemble per cell type (notebook cell 7)."""
    paths.require_path(paths.PARM_MODEL_DIR, "PARM_MODEL_DIR",
                     "score human promoters")
    from scorers.parm_promoter_score import PARMScorer
    scores = np.full(len(df), np.nan)
    for ct in sorted(df["cell_type"].unique()):
        mask = (df["cell_type"] == ct).to_numpy()
        scorer = PARMScorer(
            model_directory=paths.PARM_CELL_DIR.format(cell_type=ct),
            device=device)
        sub = df.loc[mask, "seq"].tolist()
        out = scorer.score_sequences(sub, batch_size=512)
        scores[mask] = _align_by_sequence(sub, out, "sequence", "prediction")
    if np.isnan(scores).any():
        raise RuntimeError("PARM left NaNs; a cell_type was not covered")
    return scores


# --------------------------------------------------------------- enhancer --
def _score_malinois(df, device="cuda:0", seq_len=None):
    """Malinois, matched-cell-type column, forward+revcomp averaged.

    MalinoisScorer.score_sequences already does (f(x) + f(revcomp x))/2 inside
    malinois_scoring, and its class-level _cache means the boda artifact is
    unpacked once per process no matter how many cell types we loop over.
    """
    paths.require_path(paths.MALINOIS_DIR, "MALINOIS_DIR", "score enhancers")
    paths.require_path(paths.MALINOIS_ARTIFACT, "MALINOIS_ARTIFACT",
                     "score enhancers")
    from scorers.malinois_scorer import MalinoisScorer
    scores = np.full(len(df), np.nan)
    for ct in sorted(df["cell_type"].unique()):
        mask = (df["cell_type"] == ct).to_numpy()
        scorer = MalinoisScorer(
            artifact_path=paths.MALINOIS_ARTIFACT,
            malinois_dir=paths.MALINOIS_DIR,
            cell_type_index=ct,
            seq_len=seq_len)
        sub = df.loc[mask, "seq"].tolist()
        out = scorer.score_sequences(sub, batch_size=512)
        scores[mask] = out["prediction"].to_numpy(dtype=np.float64)
    if np.isnan(scores).any():
        raise RuntimeError("Malinois left NaNs; a cell_type was not covered")
    return scores


SCORERS = {
    "reglm": _score_reglm,
    "gemorna": _score_gemorna,
    "parm": _score_parm,
    "malinois": _score_malinois,
}


def score(scorer, df, device="cuda:0", seq_len=None):
    """Score `df["seq"]` with one of the four oracles.

    `scorer` is one of SCORERS.  parm and malinois are cell-type specific and
    need a `cell_type` column.  `seq_len` is the design length, which only
    Malinois needs (it pads to its own input length around the insert).
    """
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}; known: {sorted(SCORERS)}")
    if scorer in ("parm", "malinois") and "cell_type" not in df.columns:
        raise ValueError(f"{scorer} needs a cell_type column")
    if scorer == "malinois":
        return _score_malinois(df, device=device, seq_len=seq_len)
    return SCORERS[scorer](df, device=device)


def malinois_safe_cwd():
    """boda's load_model unpacks into ./artifacts after rmtree-ing it.

    Any process that will touch Malinois must chdir somewhere private first,
    or two concurrent scorers will delete each other's artifacts mid-load.
    """
    import tempfile
    d = tempfile.mkdtemp(prefix=f"malinois_cwd_{os.getpid()}_")
    os.chdir(d)
    return d
