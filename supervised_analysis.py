import pandas as pd
import numpy as np
import torch
from collections import Counter
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

FORGET_LABELS = {"multiplication_riddle", "division_riddle", "multiplication_symbolic", "division_symbolic"}
RETAIN_LABELS = {"addition_riddle", "subtraction_riddle", "addition_symbolic", "subtraction_symbolic", "english"}


def plot_layer_concept_trends(results_dir: str):
    """
    Aggregates supervised analysis from all layers and plots trends.
    """
    results_path = Path(results_dir)
    all_data = []

    # 1. Collect data from all layer folders
    for layer_folder in sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1])):
        layer_idx = int(layer_folder.name.split('_')[1])
        json_file = layer_folder / "feature_analysis_supervised.json"

        if not json_file.exists():
            continue

        with open(json_file, 'r') as f:
            layer_results = json.load(f)

        for latent_idx, profile in layer_results.items():
            all_data.append({
                'layer': layer_idx,
                'latent_idx': int(latent_idx),
                'concept': profile['dominant_concept'],
                'scd': profile['scd_score'],
                'mean_act': profile['activation_stats']['mean'],
                'purity': profile['purity_score']
            })

    df = pd.DataFrame(all_data)

    # 2. Setup Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.set_style("whitegrid")

    # Plot A: Mean SCD per Concept across Layers
    # This shows WHERE in the model multiplication/division features become most specific
    sns.lineplot(
        data=df, x='layer', y='scd', hue='concept',
        ax=axes[0], marker='o', err_style="band", errorbar='sd'
    )
    axes[0].set_title("Concept Specificity (SCD) across Model Layers", fontsize=15)
    axes[0].set_ylabel("Mean SCD Score (Higher = More Specific)")
    axes[0].axhline(0, ls='--', color='black', alpha=0.5)

    # Plot B: Activation Intensity (Mean) per Concept
    # This shows which layers are "computing" these concepts most intensely
    sns.lineplot(
        data=df, x='layer', y='mean_act', hue='concept',
        ax=axes[1], marker='s', err_style="band", errorbar='sd'
    )
    axes[1].set_title("Activation Intensity (Mean) across Model Layers", fontsize=15)
    axes[1].set_ylabel("Mean Activation Value")

    plt.tight_layout()
    plt.show()

    # 3. Print a quick Summary Table for you
    print("\n--- Summary Statistics by Concept ---")
    summary = df.groupby('concept').agg({
        'scd': ['mean', 'std', 'max'],
        'mean_act': ['mean', 'std'],
        'latent_idx': 'count'
    }).round(3)
    print(summary)


def analyze_features_supervised(
        feature_acts: torch.Tensor,
        labels: List[str],
        sample_ids: List[int],
        token_ids: List[int],
        tokenizer,
        top_k: int = 20,
        dominance_threshold: float = 0.5,
        save_raw: bool = True,
        forget_labels: set = FORGET_LABELS,
        retain_labels: set = RETAIN_LABELS,
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

    # This maps every token activation to its sample_id and label
    base_df = pd.DataFrame({
        'sample_id': sample_ids_arr,
        'label': token_metadata
    })

    for latent_idx in range(n_latents):
        latent_activations = feature_acts_np[:, latent_idx]

        # --- SCD Calculation Logic ---
        # 1. Add current latent activations to our dataframe
        base_df['act'] = latent_activations

        # 2. Get the maximum activation per sample (sentence)
        relevant_samples = base_df[base_df['label'].isin(forget_labels | retain_labels)]
        sample_max_acts = relevant_samples.groupby(['sample_id', 'label'])['act'].max().reset_index()

        # 3. Compute mean of maximums for each group
        # a_activating: Average peak activation in 'forget' sentences
        # a_neutral: Average peak activation in 'retain' sentences
        a_activating = sample_max_acts[sample_max_acts['label'].isin(forget_labels)]['act'].mean()
        a_neutral = sample_max_acts[sample_max_acts['label'].isin(retain_labels)]['act'].mean()

        # 4. Calculate SCD (Log-Ratio)
        # Higher positive value = stronger detection of the target concept
        scd_score = np.log((a_activating + 1e-9) / (a_neutral + 1e-9))

        # --- Standard Top-K Profiling ---
        top_indices = np.argsort(latent_activations)[-top_k:][::-1]
        exemplar_labels = token_metadata[top_indices]
        exemplar_magnitudes = latent_activations[top_indices]

        semantic_counts = Counter(exemplar_labels)
        most_frequent_concept, frequent_concept_count = semantic_counts.most_common(1)[0]
        dominance_ratio = frequent_concept_count / top_k

        probs = np.array(list(semantic_counts.values())) / top_k
        entropy = -np.sum(probs * np.log2(probs + 1e-9))

        assigned_concept = most_frequent_concept if dominance_ratio >= dominance_threshold else "polysemantic"

        profile = {
            'dominant_concept': assigned_concept,
            'purity_score': round(float(dominance_ratio), 3),
            'scd_score': round(float(scd_score), 4),
            'mean_max_activating': round(float(a_activating), 4),
            'mean_max_neutral': round(float(a_neutral), 4),
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

    # --- Summary Output ---
    concept_map = {}
    for idx, p in feature_profiles.items():
        concept = p['dominant_concept']
        concept_map.setdefault(concept, []).append(idx)

    print("\nLatent-to-Concept Summary (Sorted by SCD Score):")
    for concept, indices in sorted(concept_map.items()):
        indices.sort(key=lambda x: feature_profiles[x]['scd_score'], reverse=True)
        top_indices_str = ", ".join(
            [f"{idx}(SCD:{feature_profiles[idx]['scd_score']:.2f})" for idx in indices[:5]])
        print(f"  {concept:25} | Total: {len(indices):3} | Top Latents: {top_indices_str}")

    return feature_profiles