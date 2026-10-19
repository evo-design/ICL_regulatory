#!/usr/bin/env python3
"""
Standalone script to score DNA sequences using the PARM promoter activity model.
The model predicts promoter activity (Log2RPM) from DNA sequences using a ResNet
with attention pooling architecture, averaging predictions across multiple folds.

Adapted from: https://github.com/vansteensellab/PARM

Usage:
    python parm_promoter_score.py --sequences SEQ1 SEQ2 ...
    python parm_promoter_score.py --fasta input.fasta
    python parm_promoter_score.py --random_test  # Test with random sequences
"""

import argparse
import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from einops.layers.torch import Rearrange


# ============================================================================
# Model Architecture (from PARM: PARM_utils_load_model.py)
# ============================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)
        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim=-1)
        return (x * attn).sum(dim=-1)


class ResNet_Attentionpool(nn.Module):
    def __init__(self, L_max, n_block, filter_size=125, weight_file=None,
                 type_loss='poisson', validation=False):
        super(ResNet_Attentionpool, self).__init__()

        self.type_loss = type_loss
        self.validation = validation
        self.L_max = L_max
        self.vocab = 4  # A, C, G, T

        kernel_size = 7
        stem_kernel_size = 7
        self.n_blocks = n_block

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.vocab, filter_size, stem_kernel_size, padding="same"),
            Residual(ConvBlock(filter_size)),
            AttentionPool(filter_size, pool_size=2),
        )

        # Conv tower
        conv_layers = []
        initial_filter_size = filter_size
        prev_filter_size = filter_size
        for block in range(self.n_blocks):
            if block > 4:
                filter_size = int(initial_filter_size * 0.2)
            conv_layers.append(
                nn.Sequential(
                    ConvBlock(prev_filter_size, filter_size, kernel_size=kernel_size),
                    Residual(ConvBlock(filter_size, filter_size, kernel_size=1)),
                    AttentionPool(filter_size, pool_size=2),
                )
            )
            prev_filter_size = filter_size

        self.conv_tower = nn.Sequential(*conv_layers)

        if self.type_loss == 'heteroscedastic':
            self.log_var = nn.Linear(filter_size, 1)

        self.linear1 = nn.Linear(filter_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.stem(x)
        out = self.conv_tower(out)
        out = torch.max(out, dim=-1).values
        out = out.view(out.size(0), -1)
        out = self.linear1(out)

        if self.type_loss == 'heteroscedastic':
            mu = self.linear1(out)
            log_var = self.log_var(out)
            if self.validation:
                return mu
            return mu, log_var

        if self.type_loss == 'poisson':
            out = self.relu(out)

        return out


# ============================================================================
# Model Loading (from PARM: PARM_utils_load_model.py)
# ============================================================================

def load_PARM(weight_file, L_max=600, n_block=5, filter_size=125,
              train=False, type_loss='poisson'):
    """Load a single PARM fold model from a .parm weight file."""
    model = ResNet_Attentionpool(
        L_max=L_max, n_block=n_block, filter_size=filter_size,
        weight_file=weight_file, type_loss=type_loss, validation=True
    )
    model_weights = torch.load(weight_file, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    model.eval()
    return model


# ============================================================================
# Helper Functions (from PARM: PARM_predict.py)
# ============================================================================

def sequence_to_onehot(sequences, L_max):
    """
    One-hot encode DNA sequences with center-padding to L_max.

    Args:
        sequences: list of DNA strings (A/C/G/T/N)
        L_max: maximum sequence length for padding

    Returns:
        np.array of shape (n_sequences, L_max, 4)
    """
    letter_to_vector = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]),
        "T": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0]),
    }
    one_hot = []
    for seq in sequences:
        x = np.array([letter_to_vector[s] for s in seq])
        pw = (L_max - x.shape[0]) / 2
        PW = [int(np.ceil(pw)), int(np.floor(pw))]
        one_hot.append(np.pad(x, [PW, [0, 0]], constant_values=0))
    return np.array(one_hot)


def validate_sequence(seq: str) -> bool:
    """Validate that sequence only contains valid DNA nucleotides."""
    valid_set = set('ACGTN')
    return all(base in valid_set for base in seq.upper())


def generate_random_sequence(length: int) -> str:
    """Generate a random DNA sequence."""
    return ''.join(random.choices('ACGT', k=length))


def parse_fasta(fasta_path: str) -> list:
    """Parse a FASTA file and return list of sequences."""
    sequences = []
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append(''.join(current_seq))

    return sequences


# ============================================================================
# Main Scorer Class
# ============================================================================

class PARMScorer:
    """Score DNA sequences using PARM promoter activity prediction model.

    Loads all fold models (.parm files) from a given directory and averages
    predictions across folds.
    """

    def __init__(self, model_directory: str, device: str = None,
                 filter_size: int = 125, type_loss: str = 'poisson',
                 L_max: int = 600):
        """
        Initialize the scorer.

        Args:
            model_directory: Path to directory containing .parm fold checkpoints
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
            filter_size: Filter size for the PARM model (default 125).
            type_loss: Loss type the model was trained with (default 'poisson').
            L_max: Maximum sequence length for padding (default 600).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.L_max = L_max
        self.filter_size = filter_size
        self.type_loss = type_loss

        # Discover fold files
        fold_paths = sorted([
            os.path.join(model_directory, f)
            for f in os.listdir(model_directory) if f.endswith('.parm')
        ])
        if len(fold_paths) == 0:
            raise ValueError(
                f"No .parm model files found in {model_directory}. "
                "Please check the path."
            )

        # Validate consistent model prefix
        prefixes = set(
            os.path.basename(p).split("_fold")[0] for p in fold_paths
        )
        if len(prefixes) > 1:
            raise ValueError(
                f"Multiple model prefixes found: {prefixes}. "
                "All folds should belong to the same model."
            )
        self.model_name = prefixes.pop()

        # Load all fold models
        self.models = []
        for fp in fold_paths:
            model = load_PARM(
                fp, L_max=L_max, filter_size=filter_size, type_loss=type_loss
            )
            model = model.to(self.device)
            self.models.append(model)

        print(f"Loaded {len(self.models)} PARM fold model(s) "
              f"for '{self.model_name}' from {model_directory}")
        print(f"Using device: {self.device}")

    def score_sequences(self, sequences: list, batch_size: int = 32) -> pd.DataFrame:
        """
        Score a list of DNA sequences.

        Args:
            sequences: List of DNA sequences (strings of A/C/G/T/N)
            batch_size: Batch size for inference

        Returns:
            DataFrame with columns: sequence, length, prediction
        """
        # Validate sequences
        valid_sequences = []
        for i, seq in enumerate(sequences):
            seq = seq.upper()
            if not validate_sequence(seq):
                print(f"Warning: Sequence {i} contains invalid characters, "
                      f"skipping: {seq[:50]}...")
                continue
            if len(seq) > self.L_max:
                print(f"Warning: Sequence {i} is too long "
                      f"({len(seq)} > {self.L_max}), skipping")
                continue
            valid_sequences.append(seq)

        if not valid_sequences:
            return pd.DataFrame(columns=['sequence', 'length', 'prediction'])

        # Score in batches, averaging across folds
        all_scores = []  # shape will be (n_folds, n_sequences)

        with torch.no_grad():
            for model in self.models:
                fold_scores = []
                for i in range(0, len(valid_sequences), batch_size):
                    batch_seqs = valid_sequences[i:i + batch_size]
                    onehot = sequence_to_onehot(batch_seqs, L_max=self.L_max)
                    x = torch.tensor(
                        np.float32(onehot), device=self.device
                    ).permute(0, 2, 1)  # (batch, 4, L_max)
                    scores = model(x).cpu().numpy().flatten()
                    fold_scores.append(scores)
                all_scores.append(np.concatenate(fold_scores))

        # Average across folds
        avg_scores = np.mean(all_scores, axis=0)

        results = []
        for seq, score in zip(valid_sequences, avg_scores):
            results.append({
                'sequence': seq,
                'length': len(seq),
                'prediction': float(score),
            })

        return pd.DataFrame(results)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score DNA sequences using PARM promoter activity model"
    )
    parser.add_argument('--sequences', nargs='+', type=str,
                        help='DNA sequences to score')
    parser.add_argument('--fasta', type=str,
                        help='Path to FASTA file with sequences')
    parser.add_argument('--random_test', action='store_true',
                        help='Run test with randomly generated DNA sequences')
    parser.add_argument('--num_random', type=int, default=10,
                        help='Number of random sequences to generate (default: 10)')
    parser.add_argument('--seq_length', type=int, default=231,
                        help='Length of random sequences (default: 231)')
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('PARM_CELL_DIR'),
                        required='PARM_CELL_DIR' not in os.environ,
                        help='Path to directory containing .parm fold checkpoints')
    parser.add_argument('--output', type=str,
                        help='Output file path (CSV format)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--filter_size', type=int, default=125,
                        help='PARM model filter size (default: 125)')
    parser.add_argument('--type_loss', type=str, default='poisson',
                        choices=['poisson', 'MSE', 'heteroscedastic'],
                        help='Loss type the model was trained with (default: poisson)')
    parser.add_argument('--L_max', type=int, default=600,
                        help='Maximum sequence length for padding (default: 600)')

    args = parser.parse_args()

    # Determine input sequences
    sequences = []

    if args.random_test:
        print(f"\n=== Generating {args.num_random} random DNA sequences "
              f"of length {args.seq_length} ===\n")
        random.seed(42)
        sequences = [generate_random_sequence(args.seq_length)
                     for _ in range(args.num_random)]
    elif args.fasta:
        print(f"\n=== Loading sequences from {args.fasta} ===\n")
        sequences = parse_fasta(args.fasta)
    elif args.sequences:
        sequences = args.sequences
    else:
        print("Error: No input sequences provided!")
        print("Use --sequences, --fasta, or --random_test")
        sys.exit(1)

    print(f"Processing {len(sequences)} sequences...")

    # Initialize scorer
    scorer = PARMScorer(
        model_directory=args.model_dir,
        device=args.device,
        filter_size=args.filter_size,
        type_loss=args.type_loss,
        L_max=args.L_max,
    )

    # Score sequences
    results = scorer.score_sequences(sequences, batch_size=args.batch_size)

    # Display results
    print("\n=== Results ===\n")
    print(results.to_string(index=False))

    # Summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Mean prediction: {results['prediction'].mean():.4f}")
    print(f"Std prediction:  {results['prediction'].std():.4f}")
    print(f"Min prediction:  {results['prediction'].min():.4f}")
    print(f"Max prediction:  {results['prediction'].max():.4f}")

    # Save results if output path provided
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
