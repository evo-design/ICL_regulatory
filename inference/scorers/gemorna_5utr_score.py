#!/usr/bin/env python3
"""
Script to score genome sequences using GEMORNA's 5'UTR prediction model.
The model predicts translational capacity/efficiency of 5'UTR sequences.

Usage:
    python gemorna_5utr_score.py --sequences SEQ1 SEQ2 ...
    python gemorna_5utr_score.py --fasta input.fasta
    python gemorna_5utr_score.py --random_test  # Test with random sequences
"""

import argparse
import sys
import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ============================================================================
# Model Definition (from GEMORNA)
# ============================================================================

class Model5UTR(nn.Module):
    """5'UTR prediction model from GEMORNA."""
    
    def __init__(self, embed_num=10, embed_dim=64, kernel_num=128, dropout=0.1):
        super(Model5UTR, self).__init__()
        
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.rnn_layer = nn.GRU(
            input_size=embed_dim,
            hidden_size=kernel_num,
            num_layers=2,
            bidirectional=False,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.Linear(kernel_num * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, hidden = self.rnn_layer(x)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        logit = self.decoder(hidden).squeeze(-1)
        return logit


# ============================================================================
# Helper Functions (adapted from GEMORNA)
# ============================================================================

# Vocabulary for tokenization
VOCAB = {'[PAD]': 0, 'A': 5, 'U': 6, 'G': 7, 'C': 8, 'N': 9}
MAX_SEQ_LEN = 100  # Maximum sequence length for padding

# Scaling parameters (from GEMORNA)
SCALE_MEAN = 5.19937892
SCALE_STD = 1.40592675


def validate_sequence(seq: str) -> bool:
    """Validate that sequence only contains valid nucleotides."""
    valid_set = set('ACGTUN')
    seq_upper = seq.upper()
    return all(base in valid_set for base in seq_upper)


def tokenize(seq: str) -> list:
    """Tokenize a sequence to integer indices."""
    seq = seq.upper().replace("T", "U")
    tokens = [VOCAB[c] for c in seq]
    return tokens


def scale_prediction(pred: float) -> float:
    """Scale the raw model prediction to interpretable value."""
    return pred * SCALE_STD + SCALE_MEAN


def generate_random_sequence(length: int) -> str:
    """Generate a random DNA sequence."""
    return ''.join(random.choices('ACGT', k=length))


# ============================================================================
# Main Scoring Functions
# ============================================================================

class GEMORNA5UTRScorer:
    """Score sequences using GEMORNA 5'UTR prediction model."""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the scorer.
        
        Args:
            checkpoint_path: Path to the 5utr.pt checkpoint file
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = Model5UTR().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        print(f"Loaded GEMORNA 5'UTR model from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def score_sequences(self, sequences: list, batch_size: int = 32) -> pd.DataFrame:
        """
        Score multiple sequences.
        
        Args:
            sequences: List of DNA/RNA sequences
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with columns: sequence, length, raw_score, scaled_score
        """
        results = []
        
        # Validate and prepare sequences
        valid_sequences = []
        for i, seq in enumerate(sequences):
            if not validate_sequence(seq):
                print(f"Warning: Sequence {i} contains invalid characters, skipping: {seq[:50]}...")
                continue
            if len(seq) > MAX_SEQ_LEN:
                print(f"Warning: Sequence {i} is too long ({len(seq)} > {MAX_SEQ_LEN}), truncating")
                seq = seq[:MAX_SEQ_LEN]
            valid_sequences.append(seq)
        
        if not valid_sequences:
            return pd.DataFrame(columns=['sequence', 'length', 'raw_score', 'scaled_score'])
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(valid_sequences), batch_size):
                batch_seqs = valid_sequences[i:i + batch_size]
                
                # Tokenize and pad
                tokenized_batch = []
                for seq in batch_seqs:
                    tokens = tokenize(seq)
                    # Pad to MAX_SEQ_LEN
                    padded = tokens + [VOCAB['[PAD]']] * (MAX_SEQ_LEN - len(tokens))
                    tokenized_batch.append(padded)
                
                # Convert to tensor
                batch_tensor = torch.tensor(tokenized_batch, device=self.device)
                
                # Get predictions
                raw_scores = self.model(batch_tensor).cpu().numpy()
                
                # Store results
                for seq, raw_score in zip(batch_seqs, raw_scores):
                    scaled_score = scale_prediction(raw_score)
                    results.append({
                        'sequence': seq,
                        'length': len(seq),
                        'raw_score': float(raw_score),
                        'scaled_score': float(scaled_score)
                    })
        
        return pd.DataFrame(results)


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


def main():
    parser = argparse.ArgumentParser(description="Score sequences using GEMORNA 5'UTR model")
    parser.add_argument('--sequences', nargs='+', type=str, 
                        help='Sequences to score')
    parser.add_argument('--fasta', type=str, 
                        help='Path to FASTA file with sequences')
    parser.add_argument('--random_test', action='store_true',
                        help='Run test with random sequences')
    parser.add_argument('--num_random', type=int, default=10,
                        help='Number of random sequences to generate for testing')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='Length of random sequences')
    parser.add_argument('--checkpoint', type=str, 
                        default=os.environ.get('GEMORNA_5UTR_CKPT'),
                        required='GEMORNA_5UTR_CKPT' not in os.environ,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, 
                        help='Output file path (CSV format)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Determine input sequences
    sequences = []
    
    if args.random_test:
        print(f"\n=== Generating {args.num_random} random sequences of length {args.seq_length} ===\n")
        random.seed(42)  # For reproducibility
        sequences = [generate_random_sequence(args.seq_length) for _ in range(args.num_random)]
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
    scorer = GEMORNA5UTRScorer(args.checkpoint, device=args.device)
    
    # Score sequences
    results = scorer.score_sequences(sequences, batch_size=args.batch_size)
    
    # Display results
    print("\n=== Results ===\n")
    print(results.to_string(index=False))
    
    # Summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Mean scaled score: {results['scaled_score'].mean():.4f}")
    print(f"Std scaled score: {results['scaled_score'].std():.4f}")
    print(f"Min scaled score: {results['scaled_score'].min():.4f}")
    print(f"Max scaled score: {results['scaled_score'].max():.4f}")
    
    # Save results if output path provided
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
