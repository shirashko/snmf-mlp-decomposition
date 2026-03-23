import argparse
import json
from pyexpat import features

import torch
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
torch.serialization.add_safe_globals([GemmaTokenizerFast])
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from dotenv import load_dotenv

from utils import resolve_device, set_seed
from model_utils import load_local_model, LocalModel
from experiments.snmf_interp.generate_vocab_proj import (
    get_vocab_proj_gemma_hf,
    get_vocab_proj_residual_hf,
)

load_dotenv()


def analyze_features_supervised(
        feature_acts: torch.Tensor,
        labels: List[str],
        sample_ids: List[int],
        token_ids: List[int],
        tokenizer,
        top_k: int = 20,
        dominance_threshold: float = 0.5,
        save_raw: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Performs semantic profiling of latent features by aligning peak activations
    with supervised ground-truth metadata.

    This analysis quantifies the 'monosemanticity' of learned dictionary elements.
    For each latent feature, we extract the top-k activating tokens ('exemplars')
    and map them back to their source samples to retrieve semantic labels.

    We calculate:
    1. Purity Score: The maximum probability of a single concept within the top-k exemplars.
    2. Label Entropy: A measure of polysemanticity (higher entropy indicates a feature
       representing multiple unrelated concepts).
    3. Activation Statistics: The distributional properties of the feature across the corpus.

    Args:
        feature_acts (torch.Tensor): Activation matrix of shape (n_tokens, n_latents).
        labels (List[str]): Ground-truth concept labels for each sample in the dataset.
            Shape: (n_samples,).
        sample_ids (List[int]): A mapping from token index to its parent sample index.
            Used to resolve the semantic context of individual activations.
            Shape: (n_tokens,).
        token_ids (List[int]): Integer IDs for each token in the activation trace.
        tokenizer: The model's tokenizer used for decoding exemplar tokens into text.
        top_k (int): Number of maximal activations (exemplars) used to profile each latent.
        dominance_threshold (float): The purity cutoff (0.0-1.0). Latents below this
            threshold are categorized as 'polysemantic'.
        save_raw (bool): If True, includes the raw activation magnitudes and decoded
            exemplar tokens for manual inspection.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary where each key is a latent index
            mapping to its semantic profile (dominant concept, purity, entropy, etc.).
    """
    print(f"Profiling latents (supervised, threshold={dominance_threshold})...")

    n_tokens, n_latents = feature_acts.shape
    sample_ids_arr = np.array(sample_ids)
    labels_arr = np.array(labels)
    token_metadata = labels_arr[sample_ids_arr]

    feature_profiles = {}
    feature_acts_np = feature_acts.detach().cpu().numpy()

    for latent_idx in range(n_latents):
        latent_activations = feature_acts_np[:, latent_idx]

        # Identify top exemplars
        top_indices = np.argsort(latent_activations)[-top_k:][::-1]
        exemplar_labels = token_metadata[top_indices]
        exemplar_magnitudes = latent_activations[top_indices]

        # 3. Statistical Profiling
        semantic_counts = Counter(exemplar_labels)
        most_frequent_concept, frequent_concept_count = semantic_counts.most_common(1)[0]
        dominance_ratio = frequent_concept_count / top_k

        # Calculate Label Entropy (higher = more polysemantic)
        probs = np.array(list(semantic_counts.values())) / top_k
        entropy = -np.sum(probs * np.log2(probs + 1e-9))

        assigned_concept = most_frequent_concept if dominance_ratio >= dominance_threshold else "polysemantic"

        profile = {
            'dominant_concept': assigned_concept,
            'purity_score': round(float(dominance_ratio), 3),
            'entropy': round(float(entropy), 3),
            'concept_distribution': dict(semantic_counts),
            'activation_stats': {
                'mean': round(float(np.mean(latent_activations)), 3),
                'max': round(float(np.max(latent_activations)), 3),
                'std': round(float(np.std(latent_activations)), 3)
            }
        }

        if save_raw:
            top_tids = [token_ids[i] for i in top_indices]
            profile['raw_evidence'] = {
                'tokens': tokenizer.batch_decode([[tid] for tid in top_tids]),
                'magnitudes': np.round(exemplar_magnitudes, 3).tolist(),
                'labels': exemplar_labels.tolist(),
                'sample_ids': sample_ids_arr[top_indices].tolist()
            }

        feature_profiles[latent_idx] = profile

    # Summary Output
    concept_map = {}
    for idx, p in feature_profiles.items():
        concept = p['dominant_concept']
        concept_map.setdefault(concept, []).append(idx)

    print("\nLatent-to-Concept Summary:")
    for concept, indices in sorted(concept_map.items()):
        print(
            f"  {concept:25} | Latents: {len(indices):3} | Indices: {indices[:10]}{'...' if len(indices) > 10 else ''}")

    return feature_profiles


def analyze_features_unsupervised(
        F: torch.Tensor,
        local_model: LocalModel,
        layer_idx: int,
        top_k_tokens: int = 30,
        mode: str = "mlp",
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze features by projecting to vocabulary (unsupervised).

    Args:
        F (torch.Tensor): The feature matrix of shape (d, rank) from SNMF, where d hidden dimension
                          and rank is the number of latent features.
                          F[i, j] represents the contribution of feature j to the i-th hidden dimension.
        local_model (LocalModel): The loaded model containing the tokenizer and architecture.
        layer_idx (int): The layer number corresponding to the features in F.
        top_k_tokens (int): Number of top tokens to retrieve for each feature projection.
        mode (str): Determines how to interpret the features. Options:
            - "mlp_intermediate": Projects the feature through the MLP down-projection and post-FFN layer norm.
            - "mlp": Projects the feature directly through the post-FFN layer norm.
            - "residual": Projects the raw feature vector as a residual without MLP transformations.
    """
    print(f"Analyzing features (unsupervised vocab projection, layer {layer_idx}, mode={mode})...")
    d_feat, rank = F.shape
    device = local_model.device
    feature_analysis = {}

    hf_model = local_model.model
    base_model = hf_model.model if hasattr(hf_model, "model") else hf_model

    with torch.no_grad():
        for feature_idx in range(rank):
            feature_vec = F[:, feature_idx].to(device)
            layer = base_model.layers[layer]
            if mode == "mlp_intermediate":
                down_proj = layer.mlp.down_proj
                residual_vec = down_proj(feature_vec.unsqueeze(0)).squeeze(0)
                try:
                    post_ff_ln = layer.post_feedforward_layernorm
                    concept_vector = post_ff_ln(residual_vec.unsqueeze(0)).squeeze(0)
                except AttributeError:
                    concept_vector = residual_vec
                pos_values, pos_indices = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)
            elif mode == "mlp":
                try:
                    post_ff_ln = layer.post_feedforward_layernorm
                    concept_vector = post_ff_ln(feature_vec.unsqueeze(0)).squeeze(0)
                except AttributeError:
                    concept_vector = feature_vec
                pos_values, pos_indices = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)
            else:
                pos_values, pos_indices = get_vocab_proj_residual_hf(feature_vec, hf_model, top_k_tokens, device)
                neg_values, neg_indices = get_vocab_proj_residual_hf(-feature_vec, hf_model, top_k_tokens, device)

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
    parser.add_argument("--save-raw", action="store_true", help="Include raw token data in output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    for layer_folder in sorted(results_dir.glob("layer_*")):
        layer_num = int(layer_folder.name.split("_")[1])
        factors_path = layer_folder / "snmf_factors.pt"

        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because it doesn't exist.")
            continue

        print(f"\nProcessing {layer_folder.name}...")
        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)

        F, G = checkpoint['F'], checkpoint['G']
        token_ids, sample_ids = checkpoint['token_ids'], checkpoint['sample_ids']
        labels, mode = checkpoint['labels'], checkpoint.get('mode', 'mlp_intermediate')

        supervised_results = analyze_features_supervised(
            G, labels, sample_ids, token_ids, local_model.tokenizer,
            dominance_threshold=args.dominance_threshold, save_raw=args.save_raw
        )
        with open(layer_folder / "feature_analysis_supervised.json", 'w') as f:
            json.dump(supervised_results, f, indent=2)

        if not args.skip_vocab:
            unsupervised_results = analyze_features_unsupervised(
                F=F,
                local_model=local_model,
                layer=layer_num,
                mode=mode
            )
            with open(layer_folder / "feature_analysis_unsupervised.json", 'w') as f:
                json.dump(unsupervised_results, f, indent=2)

    print(f"\nAnalysis complete. Files saved in {args.results_dir}")

if __name__ == "__main__":
    main()