"""
MalinoisScorer – thin wrapper around the standalone Malinois MPRA model.

The Malinois model outputs 280 scores per sequence (one per MPRA condition).
For in-vitro enhancer design we typically care about individual cell-type
dimensions:

    Index 0 → K562
    Index 1 → HepG2
    Index 2 → SKNSH

A class-level cache ensures the (model, flank_builder) pair is loaded at most
once per (artifact_path, seq_len) combination, even when multiple cell types
each create their own ``MalinoisScorer`` instance.
"""

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from malinois_scoring import load_malinois, score_sequences as _malinois_score

CELL_TYPE_INDEX = {"K562": 0, "HepG2": 1, "SKNSH": 2}


class MalinoisScorer:
    """Score DNA sequences using a single output dimension of the Malinois
    MPRA model.

    Parameters
    ----------
    artifact_path : str
        Path to the Malinois ``*.tar.gz`` model artifact.
    malinois_dir : str or None
        Directory containing the unpacked artifact (defaults to the
        directory of *artifact_path*).
    cell_type_index : int
        Which output column to return.  Can also be specified as a cell-type
        name (``"K562"``, ``"HepG2"``, ``"SKNSH"``).
    seq_len : int
        Expected insert length (default 200).
    device : str or None
        Unused here but accepted for interface compatibility.
    """

    _cache: dict[tuple, tuple] = {}

    def __init__(
        self,
        artifact_path: str,
        malinois_dir: Optional[str] = None,
        cell_type_index: int | str = 0,
        seq_len: int = 200,
        device: Optional[str] = None,
    ):
        self.artifact_path = artifact_path
        self.malinois_dir = malinois_dir
        self.seq_len = seq_len

        if isinstance(cell_type_index, str):
            if cell_type_index not in CELL_TYPE_INDEX:
                raise ValueError(
                    f"Unknown cell type '{cell_type_index}'. "
                    f"Valid names: {list(CELL_TYPE_INDEX)}"
                )
            self.cell_type_index = CELL_TYPE_INDEX[cell_type_index]
        else:
            self.cell_type_index = int(cell_type_index)

        self._model = None
        self._flank_builder = None

    def _load(self):
        if self._model is not None:
            return
        key = (self.artifact_path, self.seq_len)
        if key in MalinoisScorer._cache:
            self._model, self._flank_builder = MalinoisScorer._cache[key]
            return
        print(f"[MalinoisScorer] Loading model from {self.artifact_path} "
              f"(seq_len={self.seq_len}) ...")
        model, fb = load_malinois(
            self.artifact_path,
            malinois_dir=self.malinois_dir,
            seq_len=self.seq_len,
        )
        MalinoisScorer._cache[key] = (model, fb)
        self._model, self._flank_builder = model, fb

    def score_sequences(
        self, sequences: List[str], batch_size: int = 512
    ) -> pd.DataFrame:
        """Score *sequences* and return a DataFrame with a ``prediction`` column.

        Invalid sequences (non-ACGT characters) are assigned a prediction of 0.
        """
        self._load()

        valid_chars = set("ACGT")
        valid_mask = [
            all(c in valid_chars for c in s.upper()) and len(s) > 0
            for s in sequences
        ]
        valid_seqs = [s for s, v in zip(sequences, valid_mask) if v]

        predictions = np.zeros(len(sequences), dtype=np.float32)
        if valid_seqs:
            truncated = [s[: self.seq_len] for s in valid_seqs]
            all_scores = _malinois_score(
                truncated, self._model, self._flank_builder,
                batch_size=batch_size,
            )
            col_scores = all_scores[:, self.cell_type_index]
            j = 0
            for i, v in enumerate(valid_mask):
                if v:
                    predictions[i] = col_scores[j]
                    j += 1

        return pd.DataFrame({
            "sequence": sequences,
            "length": [len(s) for s in sequences],
            "prediction": predictions,
        })
