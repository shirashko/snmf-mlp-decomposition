import torch
import json
from pathlib import Path

# Expanded dictionary for general math erasure
CONCEPT_ROOTS = {
    'multiplication': ['multipl', 'times', 'product', '*', '×', 'double', 'triple'],
    'division': ['/', 'divid', 'split', 'share', 'quotient', 'half', 'quarter', 'distribut', 'ratio'],
    'arithmetic': ['=', '+', '-', 'sum', 'total', 'calc', 'result', 'equation', 'solve', 'math', 'arithmetic'],
    'digits': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
}

def generate_optimized_snmf_mask(
        model: torch.nn.Module,
        results_dir: str,
        threshold: float = 0.02,
        target_concepts: list = ["riddle", "symbolic"],
        min_token_matches: int = 3,
        purity_threshold: float = 0.35  # Increased slightly to handle digit-related noise
):
    results_path = Path(results_dir)
    snmf_mask = {}

    # Initialize mask for all model parameters
    for name, param in model.named_parameters():
        snmf_mask[name] = torch.zeros(param.shape, device="cpu")

    # Group roots for specialized matching logic
    syntax_roots = CONCEPT_ROOTS['multiplication'] + CONCEPT_ROOTS['division'] + CONCEPT_ROOTS['arithmetic']
    digit_roots = CONCEPT_ROOTS['digits']

    total_mlp_params = 0
    total_masked_params = 0
    affected_layers_count = 0

    print(f"\n[*] Generating AGGRESSIVE Math Mask (Agnostic Substring Matching)")
    print(f"{'Layer':<8} | {'Math Features':<15} | {'Neurons Targeted':<15} | {'Sparsity'}")
    print("-" * 65)

    layer_folders = sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1]))

    for layer_folder in layer_folders:
        layer_idx = int(layer_folder.name.split('_')[1])
        proj_names = ['up_proj', 'down_proj', 'gate_proj']
        layer_params = {p: f"model.layers.{layer_idx}.mlp.{p}.weight" for p in proj_names}

        if not hasattr(model.model.layers[layer_idx].mlp, 'down_proj'):
            continue

        factors = torch.load(layer_folder / "snmf_factors.pt", map_location="cpu", weights_only=False)
        F = factors['F']

        with open(layer_folder / "feature_analysis_supervised.json", "r") as f:
            supervised = json.load(f)
        with open(layer_folder / "feature_analysis_unsupervised.json", "r") as f:
            unsupervised = json.load(f)

        matching_features = set()
        for feat_idx in supervised.keys():
            feat_data = supervised[feat_idx]
            purity = feat_data.get('purity_score', 0)
            dom_concept = feat_data.get('dominant_concept', '').lower()

            # 1. Supervised: Check if concept label is math-related
            is_supervised_math = any(target.lower() in dom_concept for target in target_concepts)

            # 2. Advanced Unsupervised: Combined Logit Search
            feat_unsupervised = unsupervised.get(feat_idx, {})
            all_feat_tokens = (
                    [t.strip().lower() for t in feat_unsupervised.get('positive_tokens', [])[:15]] +
                    [t.strip().lower() for t in feat_unsupervised.get('negative_tokens', [])[:15]]
            )

            # Separate hits into syntax vs raw digits
            syntax_hits = sum(1 for root in syntax_roots if any(root in tok for tok in all_feat_tokens))
            digit_hits = sum(1 for root in digit_roots if any(root == tok for tok in all_feat_tokens))

            # HEURISTIC: A feature is math if:
            # - It has clear arithmetic syntax (min_token_matches)
            # - OR it's a mix of at least one syntax token and multiple digits
            is_unsupervised_math = (syntax_hits >= min_token_matches) or (syntax_hits >= 1 and digit_hits >= 2)

            if (is_supervised_math or is_unsupervised_math) and purity >= purity_threshold:
                matching_features.add(int(feat_idx))

        if not matching_features:
            continue

        affected_layers_count += 1
        active_neurons = set()

        # Identify top neurons for each identified math feature
        for feat_idx in matching_features:
            feature_vec = F[:, feat_idx].abs()
            k = max(1, int(threshold * len(feature_vec)))
            _, top_indices = torch.topk(feature_vec, k)
            active_neurons.update(top_indices.tolist())

        # Create binary mask for the layer
        layer_neuron_mask = torch.zeros(F.shape[0])
        for n in active_neurons:
            layer_neuron_mask[n] = 1.0

        layer_masked_count = 0
        layer_total_params = 0
        for p_key, full_name in layer_params.items():
            param_data = dict(model.named_parameters())[full_name].data
            if p_key == 'down_proj':
                m = layer_neuron_mask.unsqueeze(0).expand_as(param_data)
            else:
                m = layer_neuron_mask.unsqueeze(1).expand_as(param_data)
            snmf_mask[full_name] = m
            layer_masked_count += m.sum().item()
            layer_total_params += param_data.numel()

        total_masked_params += layer_masked_count
        total_mlp_params += layer_total_params

        sparsity = (layer_masked_count / layer_total_params) * 100
        print(f"Layer {layer_idx:<2} | {len(matching_features):<14} | {len(active_neurons):<14} | {sparsity:.2f}%")

    total_sparsity = (total_masked_params / total_mlp_params) * 100 if total_mlp_params > 0 else 0
    print(f"\n" + "=" * 50)
    print(f"FINAL SMART AGNOSTIC MASK SUMMARY")
    print(f"[*] Total Masked Params: {int(total_masked_params):,}")
    print(f"[*] Overall Sparsity:    {total_sparsity:.2f}%")
    print(f"[*] Purity Threshold:    {purity_threshold}")
    print(f"=" * 50)

    return snmf_mask