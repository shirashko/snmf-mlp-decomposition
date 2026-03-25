import torch
import json
from pathlib import Path

# Broadened target concepts for supervised search across all layers
TARGET_CONCEPTS = [
    "multiplication", "division"
]

# Vocabulary mapping for unsupervised detection
CONCEPT_TOKEN_MAP = {
    'multiplication': ['*', '×', 'times', 'multiply', 'multiplied', 'product', 'double', 'triple', 'x'],
    'division': ['/', 'divide', 'divided', 'split', 'share', 'quotient', 'half', 'quarter', 'distribute'],
}


def generate_optimized_snmf_mask(
        model: torch.nn.Module,
        results_dir: str,
        threshold: float = 0.02,  # Top 2% neurons per feature
        target_concepts: list = None,
        min_token_matches: int = 1
):
    if target_concepts is None:
        target_concepts = TARGET_CONCEPTS

    results_path = Path(results_dir)
    snmf_mask = {}

    # Initialize mask for all model parameters with zeros
    for name, param in model.named_parameters():
        snmf_mask[name] = torch.zeros(param.shape, device="cpu")

    all_target_tokens = []
    for c in CONCEPT_TOKEN_MAP.values():
        all_target_tokens.extend(c)

    total_mlp_params = 0
    total_masked_params = 0
    affected_layers_count = 0

    print(f"\n[*] Generating BALANCED Mask (Top {threshold * 100}% neurons, Min {min_token_matches} token matches)")
    print(f"{'Layer':<8} | {'Features Found':<15} | {'Neurons Targeted':<15} | {'Noise %'}")
    print("-" * 65)

    layer_folders = sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1]))

    for layer_folder in layer_folders:
        layer_idx = int(layer_folder.name.split('_')[1])
        proj_names = ['up_proj', 'down_proj', 'gate_proj']
        layer_params = {p: f"model.layers.{layer_idx}.mlp.{p}.weight" for p in proj_names}

        if not hasattr(model.model.layers[layer_idx].mlp, 'down_proj'):
            continue

        # Load SNMF factors and analysis results
        factors = torch.load(layer_folder / "snmf_factors.pt", map_location="cpu", weights_only=False)
        F = factors['F']

        with open(layer_folder / "feature_analysis_supervised.json", "r") as f:
            supervised = json.load(f)
        with open(layer_folder / "feature_analysis_unsupervised.json", "r") as f:
            unsupervised = json.load(f)

        matching_features = set()
        for feat_idx in supervised.keys():
            # A. Supervised logic: Substring search in dominant concept name
            dom_concept = supervised[feat_idx].get('dominant_concept', '').lower()
            is_supervised_math = any(target.lower() in dom_concept for target in target_concepts)

            # B. Unsupervised logic: Search target tokens in top positive logits
            pos_tokens = [t.strip().lower() for t in unsupervised.get(feat_idx, {}).get('positive_tokens', [])[:15]]
            matches = sum(1 for t in all_target_tokens if any(t in tok for tok in pos_tokens))
            is_unsupervised_math = (matches >= min_token_matches)

            if is_supervised_math or is_unsupervised_math:
                matching_features.add(int(feat_idx))

        if not matching_features:
            continue

        affected_layers_count += 1
        active_neurons = set()

        # Identify top neurons for each identified feature to build the mask
        for feat_idx in matching_features:
            feature_vec = F[:, feat_idx].abs()
            k = max(1, int(threshold * len(feature_vec)))
            _, top_indices = torch.topk(feature_vec, k)
            active_neurons.update(top_indices.tolist())

        # Generate binary neuron mask for intermediate dimension
        layer_neuron_mask = torch.zeros(F.shape[0])
        for n in active_neurons:
            layer_neuron_mask[n] = 1.0

        layer_masked_count = 0
        layer_total_params = 0

        # Apply mask to Up, Down, and Gate projections
        for p_key, full_name in layer_params.items():
            param_data = dict(model.named_parameters())[full_name].data
            if p_key == 'down_proj':
                # [Hidden, Inter] -> mask columns
                m = layer_neuron_mask.unsqueeze(0).expand_as(param_data)
            else:
                # [Inter, Hidden] -> mask rows
                m = layer_neuron_mask.unsqueeze(1).expand_as(param_data)

            snmf_mask[full_name] = m
            layer_masked_count += m.sum().item()
            layer_total_params += param_data.numel()

        total_masked_params += layer_masked_count
        total_mlp_params += layer_total_params

        sparsity = (layer_masked_count / layer_total_params) * 100
        print(f"Layer {layer_idx:<2} | {len(matching_features):<14} | {len(active_neurons):<14} | {sparsity:.2f}%")

    # --- Final Summary Reporting (Corrected outside the loop) ---
    total_sparsity = (total_masked_params / total_mlp_params) * 100 if total_mlp_params > 0 else 0

    print(f"\n" + "=" * 50)
    print(f"FINAL BALANCED MASK SUMMARY")
    print(f"[*] Total Masked Params: {int(total_masked_params):,}")
    print(f"[*] Overall Sparsity:    {total_sparsity:.2f}%")
    print(f"[*] Affected Layers:     {affected_layers_count}")
    print(f"[*] Thresholds:          Top {threshold * 100}% neurons, Min {min_token_matches} tokens")
    print(f"=" * 50)

    return snmf_mask