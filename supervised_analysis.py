from collections import Counter
from typing import List, Dict, Any
import numpy as np
import torch

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