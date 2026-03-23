"""
SNMF Runner for Local Models

Runs SNMF on a local model to discover interpretable features in MLP activations.

This script adapts the snmf-mlp-decomposition project to work with local models
(like gemma-2-0.3B_all_arithmetic+eng).

Usage:
    python -m targeted_undo.run_snmf \
        --model-path models/gemma2-2.03B_best_unlearn_model \
        --data-path data/data.json \
        --output-dir outputs/snmf_results \
        --layers 0,1,2,3,4,5 \
        --rank 50 \
        --mode mlp

Output:
    - snmf_factors.pt: The learned F and G matrices
    - snmf_analysis.json: Feature interpretations and statistics
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from model_utils import LocalModel, load_local_model
from activation_utils import LocalActivationGenerator
from data_utils.concept_dataset import SupervisedConceptDataset
from factorization.seminmf import NMFSemiNMF
from dotenv import load_dotenv
import os

from experiments.snmf_interp.generate_vocab_proj import (
    get_vocab_proj_gemma_hf,
    get_vocab_proj_residual_hf,
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("Successfully loaded HF_TOKEN from .env")
else:
    print("Warning: HF_TOKEN not found in .env file. Gated models may fail to load.")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_snmf(
        activations: torch.Tensor,
        rank: int,
        device: str = "cpu",
        sparsity: float = 0.01,
        max_iter: int = 5000,
        patience: int = 300,
        init: str = "svd",
        normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Semi-NMF on activations.

    Args:
        activations: Tensor of shape (num_tokens, d_activation)
        rank: Number of factors (features) to learn
        device: Device for computation
        sparsity: Sparsity level for features
        max_iter: Maximum iterations
        patience: Early stopping patience
        init: Initialization method ("random", "svd", "knn"). SVD is most stable.
        normalize: Whether to L2 normalize activations before SNMF

    Returns:
        (F, G) matrices where A ≈ F @ G.T
    """

    print(f"Running SNMF with rank={rank}, sparsity={sparsity}, init={init}")
    print(f"  Input shape: {activations.shape}")

    # Optional: normalize activations (helps with intermediate MLP)
    if normalize:
        norms = activations.norm(dim=1, keepdim=True).clamp_min(1e-8)
        activations = activations / norms
        print(f"  Normalized activations (L2 per sample)")

    # SNMF expects (d_features, n_samples), so transpose
    A = activations.T.to(device)

    nmf = NMFSemiNMF(rank, fitting_device=device, sparsity=sparsity)
    nmf.fit(A, max_iter=max_iter, patience=patience, verbose=True, init=init)

    F = nmf.F_.detach().cpu()  # (d_features, rank)
    G = nmf.G_.detach().cpu()  # (n_samples, rank)

    print(f"  F shape: {F.shape}, G shape: {G.shape}")

    return F, G


def _interpret_tokens(tokens: List[str]) -> str:
    """Generate a simple interpretation from top tokens."""
    tokens_str = ", ".join([f'"{t}"' for t in tokens[:5]])
    return f"Top tokens: {tokens_str}"


def parse_int_list(spec: str) -> List[int]:
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SNMF on a local model to discover interpretable features.",
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to local model directory")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to concept dataset JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results")

    # SNMF configuration
    parser.add_argument("--layers", type=str, default="0",
                        help="Layers to analyze (e.g., '0,1,2' or '0-5')")
    parser.add_argument("--rank", type=int, default=50,
                        help="Number of features to discover (k)")
    parser.add_argument("--sparsity", type=float, default=0.01,
                        help="SNMF sparsity parameter")
    parser.add_argument("--init", type=str, default="svd", choices=["random", "svd", "knn"],
                        help="SNMF initialization method (default: svd, more stable)")
    parser.add_argument("--normalize", action="store_true",
                        help="L2 normalize activations before SNMF (helps with intermediate)")
    parser.add_argument("--mode", type=str, default="mlp_intermediate",
                        choices=['mlp', 'mlp_intermediate', 'residual'],
                        help="Which activations to use: "
                             "mlp=post-down_proj (320d), "
                             "mlp_intermediate=pre-down_proj (1280d, like original SNMF repo), "
                             "residual=hidden states (320d). Default: mlp_intermediate")

    # Processing configuration
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for activation collection")
    parser.add_argument("--max-iter", type=int, default=5000,
                        help="Maximum SNMF iterations")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for computation (default: auto)")
    parser.add_argument("--seed", type=int, default=42)

    # Analysis configuration
    parser.add_argument("--dominance-threshold", type=float, default=0.5,
                        help="Min percentage (0.0-1.0) for a concept to be 'dominant'. "
                             "If no concept exceeds this, feature is labeled 'mixed'. (default: 0.5)")
    parser.add_argument("--top-k-analysis", type=int, default=20,
                        help="Number of top activating tokens to analyze per feature (default: 20)")
    parser.add_argument("--top-k-vocab", type=int, default=30,
                        help="Number of top vocabulary tokens to show in unsupervised analysis (default: 30)")
    parser.add_argument("--skip-vocab-projection", action="store_true",
                        help="Skip unsupervised vocab projection analysis (faster)")
    parser.add_argument("--save-raw", action="store_true", default=True,
                        help="Save raw token data in analysis (default: True)")
    parser.add_argument("--no-save-raw", dest="save_raw", action="store_false",
                        help="Don't save raw token data (smaller output files)")

    return parser.parse_args()


def main():
    args = parse_args()

    layers = parse_int_list(args.layers)

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SNMF Analysis")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_path}")
    print(f"  Layers: {layers}")
    print(f"  Rank: {args.rank}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print("=" * 60)

    model = load_local_model(model_path=args.model_path, device=device)

    dataset = SupervisedConceptDataset(args.data_path)

    # Get the list of tuples [(prompt1, label1), (prompt2, label2), ...]
    data = dataset.get_data()
    # Unzip the list of tuples into two separate lists
    prompts, labels = zip(*data)
    # Convert back to lists (since zip returns tuples)
    prompts = list(prompts)
    labels = list(labels)

    # Generate activations
    act_gen = LocalActivationGenerator(model, data_device="cpu", mode=args.mode)
    activations_per_layer, token_ids, sample_ids = act_gen.generate_activations(
        prompts=prompts, layers=layers, batch_size=args.batch_size
    )

    del model
    torch.cuda.empty_cache()

    # Run SNMF on each layer
    all_results = {}
    for layer_idx, layer in enumerate(layers):
        print(f"\n--- Layer {layer} ---")

        activations = activations_per_layer[layer_idx]

        F, G = run_snmf(
            activations,
            rank=args.rank,
            device=device,
            sparsity=args.sparsity,
            max_iter=args.max_iter,
            init=args.init,
            normalize=args.normalize,
        )



        torch.save({
            'F': F,
            'G': G,
            'token_ids': token_ids,
            'sample_ids': sample_ids,
        }, layer_output / "snmf_factors.pt")

        all_results[layer] = {
            'F_shape': list(F.shape),
            'G_shape': list(G.shape),
            'num_features': args.rank,
        }

    config = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'layers': layers,
        'rank': args.rank,
        'sparsity': args.sparsity,
        'mode': args.mode,
        'num_samples': len(prompts),
        'num_tokens': len(token_ids),
        'dominance_threshold': args.dominance_threshold,
        'top_k_analysis': args.top_k_analysis,
        'vocab_projection_enabled': not args.skip_vocab_projection,
        'results': all_results,
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("SNMF Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()