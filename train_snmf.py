import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from model_utils import  load_local_model
from activation_utils import LocalActivationGenerator
from data_utils.concept_dataset import SupervisedConceptDataset
from factorization.seminmf import NMFSemiNMF
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SNMF on local model activations.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="0")
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--init", type=str, default="svd")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--mode", type=str, default="mlp_intermediate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-vocab-projection", action="store_true")
    parser.add_argument("--top-k-analysis", type=int, default=20)
    parser.add_argument("--dominance-threshold", type=float, default=0.5)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    print(f"Running SNMF with rank={rank}, sparsity={sparsity}, init={init}")
    print(f"  Input shape: {activations.shape}")

    # Optional: normalize activations (helps with intermediate MLP)
    if normalize:
        norms = activations.norm(dim=1, keepdim=True).clamp_min(1e-8)
        activations = activations / norms

    # SNMF expects (d_features, n_samples), so transpose
    activation_matrix = activations.T.to(device)

    nmf = NMFSemiNMF(rank, fitting_device=device, sparsity=sparsity)
    nmf.fit(activation_matrix, max_iter=max_iter, patience=patience, verbose=True, init=init)

    feature_matrix = nmf.F_.detach().cpu()  # (d_features, rank)
    coefficient_matrix = nmf.G_.detach().cpu()  # (n_samples, rank)

    print(f"  F shape: {feature_matrix.shape}, G shape: {coefficient_matrix.shape}")

    return feature_matrix, coefficient_matrix



def main():
    args = parse_args()
    layers = parse_int_list(args.layers)

    set_seed(args.seed)

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_local_model(model_path=args.model_path, device=device)
    tokenizer = model.tokenizer # Keep reference for saving

    dataset = SupervisedConceptDataset(args.data_path)
    data = dataset.get_data()
    prompts, labels = zip(*data)
    prompts, labels = list(prompts), list(labels)

    act_gen = LocalActivationGenerator(model, data_device="cpu", mode=args.mode)
    activations_per_layer, token_ids, sample_ids = act_gen.generate_activations(
        prompts=prompts, layers=layers, batch_size=args.batch_size
    )

    # Free GPU memory for Training Phase
    del model
    torch.cuda.empty_cache()

    # 2. Training Phase
    all_results = {}
    for layer_idx, layer in enumerate(layers):
        print(f"\n--- Layer {layer} ---")
        activations = activations_per_layer[layer_idx]

        feature_matrix, coefficient_matrix = run_snmf(
            activations,
            rank=args.rank,
            device=device,
            sparsity=args.sparsity,
            max_iter=args.max_iter,
            init=args.init,
            normalize=args.normalize,
        )

        layer_output = output_dir / f"layer_{layer}"
        layer_output.mkdir(exist_ok=True)

        # Save everything for independent analysis later
        torch.save({
            'F': feature_matrix,
            'G': coefficient_matrix,
            'token_ids': token_ids,
            'sample_ids': sample_ids,
            'labels': labels,
            'tokenizer': tokenizer,
            'layer': layer,
            'mode': args.mode
        }, layer_output / "snmf_factors.pt")

        all_results[layer] = {'F_shape': list(feature_matrix.shape), 'G_shape': list(coefficient_matrix.shape)}

    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nTraining Complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()