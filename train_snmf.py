import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import  Tuple

import torch
from dotenv import load_dotenv

from model_utils import load_local_model
from activation_utils import LocalActivationGenerator
from data_utils.concept_dataset import SupervisedConceptDataset
from factorization.seminmf import NMFSemiNMF
from experiments.train.train import  parse_int_list
from utils import set_seed, resolve_device

# Load environment variables (HF_TOKEN)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def setup_logger(output_dir: Path):
    """
    Configures logging to output to both the console and a permanent log file.
    """
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File Handler (Saved to Output Directory)
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SNMF on local model activations.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local HF model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the concept JSON dataset.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save logs and factors.")
    parser.add_argument("--layers", type=str, required=True, help="Layers to process (e.g., '0-13').")
    parser.add_argument("--rank", type=int, default=50, help="Number of SNMF features (latents).")
    parser.add_argument("--sparsity", type=float, default=0.01, help="L1 sparsity penalty on features.")
    parser.add_argument("--init", type=str, default="random", choices=["random", "svd", "knn"])
    parser.add_argument("--normalize", action="store_true", help="L2 normalize activations per sample.")
    parser.add_argument("--mode", type=str, default="mlp_intermediate", help="mlp, mlp_intermediate, or residual.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_snmf(
        activations: torch.Tensor,
        rank: int,
        device: str,
        sparsity: float,
        max_iter: int,
        patience: int = 1500,
        init: str = "random",
        normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes Semi-Nonnegative Matrix Factorization on the provided activations.
    """
    logging.info(f"Running SNMF: Rank={rank}, Sparsity={sparsity}, Init={init}")
    logging.info(f"  Input Matrix Shape: {activations.shape}")

    if normalize:
        logging.info("  Applying L2 normalization to activations.")
        norms = activations.norm(dim=1, keepdim=True).clamp_min(1e-8)
        activations = activations / norms

    # Matrix A is (d_features, n_samples)
    activation_matrix = activations.T.to(device)

    nmf = NMFSemiNMF(rank, fitting_device=device, sparsity=sparsity)
    # The NMFSemiNMF.fit usually prints to stdout; you can wrap it if needed.
    nmf.fit(activation_matrix, max_iter=max_iter, patience=patience, verbose=True, init=init)

    F = nmf.F_.detach().cpu()  # (d_features, rank)
    G = nmf.G_.detach().cpu()  # (n_samples, rank)

    logging.info(f"  Factorization complete. F: {F.shape}, G: {G.shape}")
    return F, G


def main():
    args = parse_args()
    layers = parse_int_list(args.layers)
    set_seed(args.seed)

    # Initialize Output Directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Logger
    logger = setup_logger(output_dir)

    logger.info("=" * 60)
    logger.info(f"SNMF TRAINING SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Save Configuration Immediately
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Load Model and Data
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

    model = load_local_model(model_path=args.model_path, device=device)

    dataset = SupervisedConceptDataset(args.data_path)
    data = dataset.get_data()
    prompts, labels = zip(*data)
    prompts, labels = list(prompts), list(labels)

    # Activation Collection
    logger.info(f"Collecting activations for {len(prompts)} samples...")
    act_gen = LocalActivationGenerator(model, data_device="cpu", mode=args.mode)
    activations_per_layer, token_ids, sample_ids = act_gen.generate_activations(
        prompts=prompts, layers=layers, batch_size=args.batch_size
    )

    # Memory Management: Offload model before starting factorization
    logger.info("Offloading model from device to free memory for factorization.")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Factorization Phase
    for layer_idx, layer in enumerate(layers):
        logger.info(f"\n>>> Processing Layer {layer} ({layer_idx + 1}/{len(layers)}) <<<")

        activations = activations_per_layer[layer_idx]

        F, G = run_snmf(
            activations,
            rank=args.rank,
            device=device,
            sparsity=args.sparsity,
            max_iter=args.max_iter,
            init=args.init,
            normalize=args.normalize
        )

        # Save Layer results
        layer_output = output_dir / f"layer_{layer}"
        layer_output.mkdir(exist_ok=True)

        output_file = layer_output / "snmf_factors.pt"
        torch.save({
            'F': F,
            'G': G,
            'token_ids': token_ids,
            'sample_ids': sample_ids,
            'labels': labels,
            'layer': layer,
            'mode': args.mode,
            'config': vars(args)
        }, output_file)

        logger.info(f"Factors for Layer {layer} saved to {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info("SNMF Training Session Successfully Completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


"""
python train_snmf.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --data-path "data/data_subsampled.json" \
    --layers "0-13" \
    --rank 256 \
    --sparsity 0.08 \
    --init "svd" \
    --normalize \
    --max-iter 10000 \
    --batch-size 8 \
    --output-dir "./final_run_all_layers"
    
python train_snmf.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --data-path "data/data_subsampled.json" \
    --layers "0-13" \
    --rank 100 \
    --init "svd" \
    --normalize \
    --max-iter 10000 \
    --batch-size 8 \
    --output-dir "./final_run_all_layers_02"
    
python train_snmf.py \
    --model-path "models/gemma2-2.03B_pretrained" \
    --data-path "data/data_subsampled.json" \
    --layers "0-13" \
    --rank 256 \
    --sparsity 0.08 \
    --init "svd" \
    --normalize \
    --max-iter 5000 \
    --batch-size 15 \
    --output-dir "./pretrained_results"
"""
