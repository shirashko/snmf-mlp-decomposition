import argparse
import json
import torch
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
torch.serialization.add_safe_globals([GemmaTokenizerFast])
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from model_utils import load_local_model, LocalModel
from dotenv import load_dotenv
from utils import resolve_device, set_seed

from experiments.snmf_interp.generate_vocab_proj import (
    get_vocab_proj_gemma_hf,
    get_vocab_proj_residual_hf,
)

load_dotenv()


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
    print(f"Analyzing features (supervised, threshold={dominance_threshold})...")

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

    print("Feature-to-concept mapping (supervised):")
    for concept, features in sorted(concept_features.items()):
        print(f"  {concept}: features {features}")

    return feature_analysis


def analyze_features_unsupervised(
        F: torch.Tensor,
        local_model: LocalModel,
        layer: int,
        top_k_tokens: int = 30,
        mode: str = "mlp",
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze features by projecting to vocabulary (unsupervised).

    Fixed for Gemma 2 architecture access.
    """
    print(f"Analyzing features (unsupervised vocab projection, layer {layer}, mode={mode})...")

    d_feat, rank = F.shape
    device = local_model.device
    feature_analysis = {}

    # Get the actual transformer backbone (Gemma2Model)
    # hf_model is usually Gemma2ForCausalLM, which has a .model attribute
    hf_model = local_model.model
    if hasattr(hf_model, "model"):
        base_model = hf_model.model
    else:
        base_model = hf_model

    with torch.no_grad():
        for feature_idx in range(rank):
            feature_vec = F[:, feature_idx].to(device)

            if mode == "mlp_intermediate":
                # Intermediate MLP activations (d_mlp=3072 in your logs)
                # Pathway: down_proj -> post_ff_ln (if exists) -> final_norm -> unembed
                down_proj = base_model.layers[layer].mlp.down_proj
                residual_vec = down_proj(feature_vec.unsqueeze(0)).squeeze(0)

                try:
                    # Gemma 2 often uses post_feedforward_layernorm
                    post_ff_ln = base_model.layers[layer].post_feedforward_layernorm
                    concept_vector = post_ff_ln(residual_vec.unsqueeze(0)).squeeze(0)
                except AttributeError:
                    concept_vector = residual_vec

                pos_values, pos_indices = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)

            elif mode == "mlp":
                # MLP output mode (post-down_proj, d_model=768)
                try:
                    post_ff_ln = base_model.layers[layer].post_feedforward_layernorm
                    concept_vector = post_ff_ln(feature_vec.unsqueeze(0)).squeeze(0)
                except AttributeError:
                    concept_vector = feature_vec

                pos_values, pos_indices = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)

            else:
                # Residual mode: feature is already in hidden space
                pos_values, pos_indices = get_vocab_proj_residual_hf(feature_vec, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_residual_hf(-feature_vec, hf_model, top_k_tokens, device)

            # Decode tokens using the tokenizer in the container
            pos_tokens = [local_model.tokenizer.decode([idx.item()]) for idx in pos_indices]
            neg_tokens = [local_model.tokenizer.decode([idx.item()]) for idx in neg_indices]

            feature_analysis[feature_idx] = {
                'positive_tokens': pos_tokens,
                'positive_printits': pos_values.cpu().tolist(),
                'negative_tokens': neg_tokens,
                'negative_printits': (-neg_values).cpu().tolist(),
                'interpretation': _interpret_tokens(pos_tokens[:10]),
            }

    print("Feature vocabulary projections:")
    for feat_idx in range(min(rank, 5)):
        tokens = feature_analysis[feat_idx]['positive_tokens'][:5]
        print(f"  Feature {feat_idx}: {tokens}")

    return feature_analysis


def _interpret_tokens(tokens: List[str]) -> str:
    """Generate a simple interpretation from top tokens."""
    tokens_str = ", ".join([f'"{t}"' for t in tokens[:5]])
    return f"Top tokens: {tokens_str}"


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-trained SNMF results.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True, help="Path to folder containing layer_X subfolders")
    parser.add_argument("--dominance-threshold", type=float, default=0.5)
    parser.add_argument("--skip-vocab", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)

    # Load model once for tokenizer and weights
    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    # Iterate over existing layer results
    for layer_folder in sorted(results_dir.glob("layer_*")):
        layer_num = int(layer_folder.name.split("_")[1])
        factors_path = layer_folder / "snmf_factors.pt"

        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because it doesn't exist.")
            continue

        print(f"\nProcessing {layer_folder.name}...")
        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)

        F = checkpoint['F']
        G = checkpoint['G']
        token_ids = checkpoint['token_ids']
        sample_ids = checkpoint['sample_ids']
        labels = checkpoint['labels']
        mode = checkpoint.get('mode', 'mlp_intermediate')

        supervised_results = analyze_features_supervised(
            G, labels, sample_ids, token_ids, local_model.tokenizer,
            dominance_threshold=args.dominance_threshold
        )

        with open(layer_folder / "feature_analysis_supervised.json", 'w') as f:
            json.dump(supervised_results, f, indent=2)

        # 2. Unsupervised Analysis
        if not args.skip_vocab:
            unsupervised_results = analyze_features_unsupervised(
                F=F, local_model=local_model.model, layer=layer_num, mode=mode
            )
            with open(layer_folder / "feature_analysis_unsupervised.json", 'w') as f:
                json.dump(unsupervised_results, f, indent=2)

    print(f"\nAnalysis complete. Files saved in {args.results_dir}")


if __name__ == "__main__":
    main()