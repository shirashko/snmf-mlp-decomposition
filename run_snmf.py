"""
SNMF Runner for Local Models

Runs Semi-Nonnegative Matrix Factorization (SNMF) on a local model to discover
interpretable features in MLP activations.

This script adapts the snmf-mlp-decomposition project to work with local models
(like your gemma-2-0.1B_all_arithmetic+eng).

Usage:
    python -m targeted_undo.run_snmf \
        --model-path gemma-2-0.1B_all_arithmetic+eng/final_model \
        --data-path src/snmf-mlp-decomposition/data/arithmetic.json \
        --output-dir outputs/snmf_results \
        --layers 0,1,2,3,4,5 \
        --rank 50 \
        --mode mlp

Output:
    - snmf_factors.pt: The learned F and G matrices
    - snmf_analysis.json: Feature interpretations and statistics
"""

import sys
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm
from model_utils import LocalModel, load_local_model

# Add project paths for SNMF repo imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SNMF_REPO_PATH = PROJECT_ROOT
SNMF_EXPERIMENTS_PATH = SNMF_REPO_PATH / "experiments"

if str(SNMF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(SNMF_REPO_PATH))
if str(SNMF_EXPERIMENTS_PATH) not in sys.path:
    sys.path.insert(0, str(SNMF_EXPERIMENTS_PATH))


# ------------------------------
# Logging Helper
# ------------------------------
def log(txt: str) -> None:
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {txt}", flush=True)


# ------------------------------
# Seed Setting
# ------------------------------
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# Dataset Loading
# ------------------------------
def load_concept_dataset(data_path: str) -> Tuple[List[str], List[str]]:
    """
    Load concept dataset in SNMF format.

    Returns:
        (prompts, labels) - Lists of prompts and their concept labels
    """
    log(f"Loading dataset from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    labels = []

    for concept, texts in data.items():
        for text in texts:
            prompts.append(text)
            labels.append(concept)

    log(f"Loaded {len(prompts)} samples across {len(data)} concepts")
    return prompts, labels


# ------------------------------
# Activation Collection (HuggingFace-based)
# ------------------------------
class LocalActivationGenerator:
    """
    Activation generator for local models using HuggingFace.

    Supports:
    - 'mlp': MLP output activations (what SNMF paper uses)
    - 'residual': Residual stream after each layer
    """

    def __init__(
            self,
            local_model: LocalModel,
            data_device: str = "cpu",
            mode: str = "mlp"
    ):
        self.local_model = local_model
        self.model = local_model.model
        self.tokenizer = local_model.tokenizer
        self.data_device = data_device
        self._mode = mode

        if mode not in ['mlp', 'mlp_intermediate', 'residual']:
            raise ValueError(f"Mode '{mode}' not supported. Use 'mlp', 'mlp_intermediate', or 'residual'.")

    def _get_mlp_module(self, layer: int):
        """Get the MLP module for a given layer (Gemma-2 specific)."""
        # Gemma-2 structure: model.layers[layer].mlp
        return self.model.model.layers[layer].mlp

    def _get_down_proj(self, layer: int):
        """Get the down_proj layer for capturing intermediate activations (Gemma-2 specific)."""
        # Gemma-2 structure: model.layers[layer].mlp.down_proj
        return self.model.model.layers[layer].mlp.down_proj

    def generate_activations(
            self,
            prompts: List[str],
            layers: List[int],
            batch_size: int = 4,
    ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """
        Generate activations for multiple layers using HuggingFace.

        Returns:
            (activations_per_layer, token_ids, sample_ids_per_token)
        """
        log(f"Generating activations for layers {layers}...")
        log(f"  Mode: {self._mode}")

        all_layer_acts = [[] for _ in layers]
        all_token_ids = []
        all_sample_ids = []

        pad_token_id = self.tokenizer.pad_token_id

        num_batches = (len(prompts) + batch_size - 1) // batch_size

        # For MLP modes, we need to register hooks
        if self._mode in ['mlp', 'mlp_intermediate']:
            # Storage for hook outputs
            mlp_outputs = {}
            hooks = []

            if self._mode == 'mlp':
                # Hook on MLP output (post-down_proj, 320 dim)
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        mlp_outputs[layer_idx] = output.detach()

                    return hook_fn

                for layer_idx, layer in enumerate(layers):
                    mlp_module = self._get_mlp_module(layer)
                    hook = mlp_module.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)

            else:  # mlp_intermediate
                # Hook on down_proj INPUT to get intermediate activations (1280 dim)
                # This is what the original SNMF repo uses
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        # input is a tuple, first element is the intermediate activations
                        mlp_outputs[layer_idx] = input[0].detach()

                    return hook_fn

                for layer_idx, layer in enumerate(layers):
                    down_proj = self._get_down_proj(layer)
                    hook = down_proj.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)

        try:
            with torch.no_grad():
                for batch_idx in tqdm(range(num_batches), desc="Collecting activations"):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(prompts))
                    batch_prompts = prompts[start:end]

                    # Tokenize with left padding
                    encoded = self.tokenizer(
                        batch_prompts,
                        padding=True,
                        padding_side="left",
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = encoded["input_ids"].to(self.local_model.device)
                    attention_mask = encoded["attention_mask"].to(self.local_model.device)

                    # Clear MLP outputs from previous batch
                    if self._mode in ['mlp', 'mlp_intermediate']:
                        mlp_outputs.clear()

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=(self._mode == 'residual'),
                    )

                    # Mask for non-padding tokens
                    mask = attention_mask.bool()

                    # Get token IDs for this batch
                    for i, prompt_idx in enumerate(range(start, end)):
                        prompt_mask = mask[i]
                        prompt_tokens = input_ids[i][prompt_mask]
                        all_token_ids.extend(prompt_tokens.cpu().tolist())
                        all_sample_ids.extend([prompt_idx] * prompt_mask.sum().item())

                    # Extract activations per layer
                    if self._mode in ['mlp', 'mlp_intermediate']:
                        # Use hook outputs
                        # mlp: (batch, seq, d_model=320) - post-down_proj
                        # mlp_intermediate: (batch, seq, d_mlp=1280) - pre-down_proj
                        for layer_idx, layer in enumerate(layers):
                            if layer_idx in mlp_outputs:
                                acts = mlp_outputs[layer_idx]
                                nonpad_acts = acts[mask].view(-1, acts.size(-1))
                                all_layer_acts[layer_idx].append(nonpad_acts.cpu())
                            else:
                                log(f"  Warning: No MLP output captured for layer {layer}")
                    else:
                        # Use hidden_states for residual mode
                        hidden_states = outputs.hidden_states
                        for layer_idx, layer in enumerate(layers):
                            if layer + 1 < len(hidden_states):
                                acts = hidden_states[layer + 1].detach()
                                nonpad_acts = acts[mask].view(-1, acts.size(-1))
                                all_layer_acts[layer_idx].append(nonpad_acts.cpu())
                            else:
                                log(f"  Warning: Layer {layer} not available")

                    # Clear memory
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            # Remove hooks
            if self._mode in ['mlp', 'mlp_intermediate']:
                for hook in hooks:
                    hook.remove()

        # Concatenate
        final_acts = [torch.cat(layer_acts, dim=0) if layer_acts else torch.empty(0)
                      for layer_acts in all_layer_acts]

        for i, layer in enumerate(layers):
            log(f"  Layer {layer}: {final_acts[i].shape}")

        return final_acts, all_token_ids, all_sample_ids


# ------------------------------
# SNMF Factorization
# ------------------------------
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
    # Import SNMF from the vendored package
    from factorization.seminmf import NMFSemiNMF

    log(f"Running SNMF with rank={rank}, sparsity={sparsity}, init={init}")
    log(f"  Input shape: {activations.shape}")

    # Optional: normalize activations (helps with intermediate MLP)
    if normalize:
        norms = activations.norm(dim=1, keepdim=True).clamp_min(1e-8)
        activations = activations / norms
        log(f"  Normalized activations (L2 per sample)")

    # SNMF expects (d_features, n_samples), so transpose
    A = activations.T.to(device)

    nmf = NMFSemiNMF(rank, fitting_device=device, sparsity=sparsity)
    nmf.fit(A, max_iter=max_iter, patience=patience, verbose=True, init=init)

    F = nmf.F_.detach().cpu()  # (d_features, rank)
    G = nmf.G_.detach().cpu()  # (n_samples, rank)

    log(f"  F shape: {F.shape}, G shape: {G.shape}")

    return F, G


# ------------------------------
# Feature Analysis (Supervised - with labels)
# ------------------------------
def analyze_features_supervised(
        G: torch.Tensor,
        labels: List[str],
        sample_ids: List[int],
        token_ids: List[int],
        tokenizer,
        top_k: int = 20,
        dominance_threshold: float = 0.5,
        save_raw: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze what concepts each SNMF feature captures using labeled data.

    Args:
        G: Activation weights matrix (n_tokens, rank)
        labels: List of concept labels for each sample
        sample_ids: Mapping from token index to sample index
        token_ids: Token IDs for each activation
        tokenizer: Tokenizer for decoding token IDs
        top_k: Number of top activating tokens to analyze
        dominance_threshold: Minimum percentage (0.0-1.0) to assign dominant concept.
                            If no concept exceeds this, labeled as "mixed".
        save_raw: Whether to include raw token data in output

    Returns:
        Dictionary mapping feature index to analysis results
    """
    log(f"Analyzing features (supervised, threshold={dominance_threshold})...")

    n_tokens, rank = G.shape

    # Map token indices to their labels
    token_labels = [labels[sample_ids[i]] for i in range(n_tokens)]

    feature_analysis = {}

    for feature_idx in range(rank):
        feature_activations = G[:, feature_idx].numpy()

        # Get top activating tokens
        top_indices = np.argsort(feature_activations)[-top_k:][::-1]
        top_activations = feature_activations[top_indices]
        top_labels = [token_labels[i] for i in top_indices]

        # Count concept distribution in top activations
        concept_counts = {}
        for label in top_labels:
            concept_counts[label] = concept_counts.get(label, 0) + 1

        # Find dominant concept with threshold
        total_top = len(top_labels)
        max_concept = max(concept_counts, key=concept_counts.get)
        max_count = concept_counts[max_concept]
        max_percentage = max_count / total_top

        if max_percentage >= dominance_threshold:
            dominant_concept = max_concept
        else:
            dominant_concept = "mixed"

        result = {
            'dominant_concept': dominant_concept,
            'dominant_percentage': float(max_percentage),
            'concept_distribution': concept_counts,
            'mean_activation': float(np.mean(feature_activations)),
            'max_activation': float(np.max(feature_activations)),
            'top_concepts': list(concept_counts.keys())[:5],
        }

        # Add raw findings if requested
        if save_raw:
            # Decode top tokens
            top_token_ids = [token_ids[i] for i in top_indices]
            top_token_texts = [tokenizer.decode([tid]) for tid in top_token_ids]
            top_sample_ids = [sample_ids[i] for i in top_indices]

            result['raw'] = {
                'top_token_indices': top_indices.tolist(),
                'top_token_ids': top_token_ids,
                'top_token_texts': top_token_texts,
                'top_activations': top_activations.tolist(),
                'top_labels': top_labels,
                'top_sample_ids': top_sample_ids,
            }

        feature_analysis[feature_idx] = result

    # Summarize
    concept_features = {}
    for feat_idx, analysis in feature_analysis.items():
        concept = analysis['dominant_concept']
        if concept not in concept_features:
            concept_features[concept] = []
        concept_features[concept].append(feat_idx)

    log("Feature-to-concept mapping (supervised):")
    for concept, features in sorted(concept_features.items()):
        log(f"  {concept}: features {features}")

    return feature_analysis


# ------------------------------
# Feature Analysis (Unsupervised - vocab projection)
# Uses functions from SNMF repo: src/snmf-mlp-decomposition/experiments/snmf_interp/generate_vocab_proj.py
# ------------------------------
from experiments.snmf_interp.generate_vocab_proj import (
    get_concept_vector_gemma_hf,
    get_vocab_proj_gemma_hf,
    get_vocab_proj_residual_hf,
)


def analyze_features_unsupervised(
        F: torch.Tensor,
        local_model: LocalModel,
        layer: int,
        top_k_tokens: int = 30,
        mode: str = "mlp",
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze features by projecting to vocabulary (unsupervised, like original SNMF repo).

    Mode determines the activation space:
    - "mlp": Post-down_proj activations (d_model=320)
    - "mlp_intermediate": Pre-down_proj activations (d_mlp=1280) - like original SNMF repo
    - "residual": Hidden states (d_model=320)

    Args:
        F: Feature directions matrix
           - For mlp/residual: (d_model, rank)
           - For mlp_intermediate: (d_mlp, rank)
        local_model: The loaded model container
        layer: Which layer these features are from
        top_k_tokens: Number of top vocabulary tokens to show
        mode: "mlp", "mlp_intermediate", or "residual"

    Returns:
        Dictionary mapping feature index to vocab projection results
    """
    log(f"Analyzing features (unsupervised vocab projection, layer {layer}, mode={mode})...")

    d_feat, rank = F.shape
    device = local_model.device

    feature_analysis = {}

    # Get the HuggingFace model for vocab projection
    hf_model = local_model.model

    with torch.no_grad():
        for feature_idx in range(rank):
            feature_vec = F[:, feature_idx].to(device)

            if mode == "mlp_intermediate":
                # Intermediate MLP activations (d_mlp=1280) - like original SNMF repo
                # Need to: down_proj → post_ff_ln → final_norm → unembed
                down_proj = hf_model.model.layers[layer].mlp.down_proj
                residual_vec = down_proj(feature_vec.unsqueeze(0)).squeeze(0)  # (d_model,)

                try:
                    post_ff_ln = hf_model.model.layers[layer].post_feedforward_layernorm
                    concept_vector = post_ff_ln(residual_vec.unsqueeze(0)).squeeze(0)
                except AttributeError:
                    concept_vector = residual_vec

                pos_values, pos_indices = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)

            elif mode == "mlp":
                # MLP output mode (post-down_proj, d_model=320)
                # Need to: post_ff_ln → final_norm → unembed
                try:
                    post_ff_ln = hf_model.model.layers[layer].post_feedforward_layernorm
                    concept_vector = post_ff_ln(feature_vec.unsqueeze(0)).squeeze(0)
                except AttributeError:
                    concept_vector = feature_vec

                pos_values, pos_indices = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)
            else:
                # Residual mode: feature is already in hidden space, just apply final norm and unembed
                pos_values, pos_indices = get_vocab_proj_residual_hf(feature_vec, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_residual_hf(-feature_vec, hf_model, top_k_tokens, device)

            # Decode tokens
            pos_tokens = [local_model.tokenizer.decode([idx.item()]) for idx in pos_indices]
            neg_tokens = [local_model.tokenizer.decode([idx.item()]) for idx in neg_indices]

            feature_analysis[feature_idx] = {
                'positive_tokens': pos_tokens,
                'positive_logits': pos_values.cpu().tolist(),
                'negative_tokens': neg_tokens,
                'negative_logits': (-neg_values).cpu().tolist(),
                'interpretation': _interpret_tokens(pos_tokens[:10]),
            }

    # Log summary
    log("Feature vocabulary projections:")
    for feat_idx in range(min(rank, 5)):  # Show first 5
        tokens = feature_analysis[feat_idx]['positive_tokens'][:5]
        log(f"  Feature {feat_idx}: {tokens}")
    if rank > 5:
        log(f"  ... and {rank - 5} more features")

    return feature_analysis


def _interpret_tokens(tokens: List[str]) -> str:
    """Generate a simple interpretation from top tokens."""
    tokens_str = ", ".join([f'"{t}"' for t in tokens[:5]])
    return f"Top tokens: {tokens_str}"


# ------------------------------
# Main Entry Point
# ------------------------------
def main():
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
                        help="Number of SNMF factors/features (default: 50)")
    parser.add_argument("--sparsity", type=float, default=0.01,
                        help="SNMF sparsity parameter (default: 0.01)")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

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

    args = parser.parse_args()

    # Parse layers
    layers = []
    for chunk in args.layers.split(','):
        if '-' in chunk:
            a, b = chunk.split('-')
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(chunk))
    layers = sorted(set(layers))

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("SNMF Analysis")
    log("=" * 60)
    log(f"  Model: {args.model_path}")
    log(f"  Data: {args.data_path}")
    log(f"  Layers: {layers}")
    log(f"  Rank: {args.rank}")
    log(f"  Mode: {args.mode}")
    log(f"  Device: {device}")
    log("=" * 60)

    # Load model
    model = load_local_model(args.model_path, device=device)

    # Load dataset
    prompts, labels = load_concept_dataset(args.data_path)

    # Generate activations
    act_gen = LocalActivationGenerator(model, data_device="cpu", mode=args.mode)
    activations_per_layer, token_ids, sample_ids = act_gen.generate_activations(
        prompts, layers, batch_size=args.batch_size
    )

    # Run SNMF for each layer
    all_results = {}

    for layer_idx, layer in enumerate(layers):
        log(f"\n--- Layer {layer} ---")

        activations = activations_per_layer[layer_idx]

        # Run SNMF
        F, G = run_snmf(
            activations,
            rank=args.rank,
            device=device,
            sparsity=args.sparsity,
            max_iter=args.max_iter,
            init=args.init,
            normalize=args.normalize,
        )

        # Analyze features (supervised - with labels)
        supervised_analysis = analyze_features_supervised(
            G, labels, sample_ids,
            token_ids=token_ids,
            tokenizer=model.tokenizer,
            top_k=args.top_k_analysis,
            dominance_threshold=args.dominance_threshold,
            save_raw=args.save_raw,
        )

        # Analyze features (unsupervised - vocab projection)
        if not args.skip_vocab_projection:
            unsupervised_analysis = analyze_features_unsupervised(
                F, model, layer,
                top_k_tokens=args.top_k_vocab,
                mode=args.mode,
            )
        else:
            unsupervised_analysis = None

        # Save layer results
        layer_output = output_dir / f"layer_{layer}"
        layer_output.mkdir(exist_ok=True)

        torch.save({
            'F': F,
            'G': G,
            'token_ids': token_ids,
            'sample_ids': sample_ids,
        }, layer_output / "snmf_factors.pt")

        # Save supervised analysis
        with open(layer_output / "feature_analysis_supervised.json", 'w') as f:
            json.dump(supervised_analysis, f, indent=2)

        # Save unsupervised analysis
        if unsupervised_analysis:
            with open(layer_output / "feature_analysis_unsupervised.json", 'w') as f:
                json.dump(unsupervised_analysis, f, indent=2)

        all_results[layer] = {
            'F_shape': list(F.shape),
            'G_shape': list(G.shape),
            'num_features': args.rank,
        }

        log(f"  Saved to {layer_output}")

    # Save overall config
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

    log("\n" + "=" * 60)
    log("SNMF Analysis Complete!")
    log(f"Results saved to: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()