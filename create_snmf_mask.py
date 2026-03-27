import torch
import json
import random
from pathlib import Path

# Dictionary of roots to identify mathematical concepts in vocabulary projections
CONCEPT_ROOTS = {
    'multiplication': ['multipl', 'times', 'product', '*', '×', 'double', 'triple'],
    'division': ['/', 'divid', 'split', 'share', 'quotient', 'half', 'quarter', 'distribut', 'ratio'],
    'arithmetic': ['=', '+', '-', 'sum', 'total', 'calc', 'result', 'equation', 'solve', 'math', 'arithmetic'],
    'digits': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
}



def generate_optimized_snmf_mask(
        model: torch.nn.Module,
        results_dir: str,
        threshold: float = 0.3,
        target_concepts: list = ["multiplication", "division"],
        min_token_matches: int = 3,
        purity_threshold: float = 0.2,
        target_projections: list = ["up_proj", "down_proj", "gate_proj"]
):
    """
    Analyzes SNMF results and generates a binary mask targeting specific math neurons.
    Selective masking of MLP components (up, down, gate) is supported via target_projections.
    """
    results_path = Path(results_dir)
    snmf_mask = {}

    # 1. Initialize zero masks for all model parameters on CPU
    for name, param in model.named_parameters():
        snmf_mask[name] = torch.zeros(param.shape, device="cpu")

    syntax_roots = CONCEPT_ROOTS['multiplication'] + CONCEPT_ROOTS['division'] + CONCEPT_ROOTS['arithmetic']
    digit_roots = CONCEPT_ROOTS['digits']

    total_mlp_params = 0
    total_masked_params = 0

    print(f"\n[*] Generating Targeted Math Mask")
    print(f"[*] Targeting components: {target_projections}")
    print(f"{'Layer':<8} | {'Math Features':<15} | {'Neurons Targeted':<15} | {'Sparsity'}")
    print("-" * 75)

    # Sort layer folders to process them in order
    layer_folders = sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1]))

    for layer_folder in layer_folders:
        layer_idx = int(layer_folder.name.split('_')[1])

        # Build the list of parameters to mask for the current layer based on user selection
        active_proj_names = [p for p in ['up_proj', 'down_proj', 'gate_proj'] if p in target_projections]
        layer_params = {p: f"model.layers.{layer_idx}.mlp.{p}.weight" for p in active_proj_names}

        # Validate that the layer structure exists
        if not hasattr(model.model.layers[layer_idx].mlp, 'down_proj'):
            continue

        # Load SNMF weights (F matrix) and analysis metadata
        factors = torch.load(layer_folder / "snmf_factors.pt", map_location="cpu", weights_only=False)
        F = factors['F']  # Feature matrix shape: (intermediate_size, rank)

        with open(layer_folder / "feature_analysis_supervised.json", "r") as f:
            supervised = json.load(f)
        with open(layer_folder / "feature_analysis_unsupervised.json", "r") as f:
            unsupervised = json.load(f)

        # 2. Identify math-related features using supervised and unsupervised heuristics
        matching_features = set()
        for feat_idx in supervised.keys():
            feat_data = supervised[feat_idx]
            purity = feat_data.get('purity_score', 0)
            dom_concept = feat_data.get('dominant_concept', '').lower()

            # Supervised check
            is_supervised_math = any(target.lower() in dom_concept for target in target_concepts)

            # Unsupervised check (token-based search)
            feat_unsupervised = unsupervised.get(feat_idx, {})
            all_feat_tokens = (
                    [t.strip().lower() for t in feat_unsupervised.get('positive_tokens', [])[:40]] +
                    [t.strip().lower() for t in feat_unsupervised.get('negative_tokens', [])[:40]]
            )

            syntax_hits = sum(1 for root in syntax_roots if any(root in tok for tok in all_feat_tokens))
            digit_hits = sum(1 for root in digit_roots if any(root == tok for tok in all_feat_tokens))

            is_unsupervised_math = (syntax_hits >= min_token_matches) or (syntax_hits >= 1 and digit_hits >= 2)

            if (is_supervised_math or is_unsupervised_math) and purity >= purity_threshold:
                matching_features.add(int(feat_idx))

        if not matching_features:
            continue

        # 3. Identify the most influential neurons (indices) for the selected features
        active_neurons = set()
        for feat_idx in matching_features:
            feature_vec = F[:, feat_idx].abs()
            k = max(1, int(threshold * len(feature_vec)))
            _, top_indices = torch.topk(feature_vec, k)
            active_neurons.update(top_indices.tolist())

        # Create a 1D neuron mask for the Intermediate Dimension
        layer_neuron_mask = torch.zeros(F.shape[0])
        for n in active_neurons:
            layer_neuron_mask[n] = 1.0

        layer_masked_count = 0
        layer_total_params_in_selected = 0

        # 4. Apply the mask specifically to selected projections (up, down, or gate)
        for p_key, full_name in layer_params.items():
            param_data = dict(model.named_parameters())[full_name].data

            if p_key == 'down_proj':
                # For down_proj [Hidden, Inter], neurons correspond to columns
                m = layer_neuron_mask.unsqueeze(0).expand_as(param_data)
            else:
                # For gate/up_proj [Inter, Hidden], neurons correspond to rows
                m = layer_neuron_mask.unsqueeze(1).expand_as(param_data)

            snmf_mask[full_name] = m
            layer_masked_count += m.sum().item()
            layer_total_params_in_selected += param_data.numel()

        total_masked_params += layer_masked_count
        total_mlp_params += layer_total_params_in_selected

        sparsity = (
                               layer_masked_count / layer_total_params_in_selected) * 100 if layer_total_params_in_selected > 0 else 0
        print(f"Layer {layer_idx:<2} | {len(matching_features):<14} | {len(active_neurons):<14} | {sparsity:.2f}%")

    total_sparsity = (total_masked_params / total_mlp_params) * 100 if total_mlp_params > 0 else 0
    print(f"\n" + "=" * 60)
    print(f"FINAL MASK SUMMARY")
    print(f"[*] Total Masked Params: {int(total_masked_params):,}")
    print(f"[*] Overall Sparsity:    {total_sparsity:.2f}% (relative to targeted components)")
    print(f"=" * 60)

    return snmf_mask


def generate_random_matching_mask(original_mask, model_config, mode="global",
                                  target_projections=["up_proj", "down_proj", "gate_proj"]):
    """
    Generates a random baseline mask.
    In 'global' mode, it redistributes the total budget of masked neurons
    across all available MLP slots defined in the model_config.
    """
    random_mask = {}
    num_layers = model_config.num_hidden_layers
    intermediate_size = model_config.intermediate_size
    hidden_size = model_config.hidden_size

    # 1. Calculate total neuron budget from the original mask
    total_neuron_budget = 0
    for k, v in original_mask.items():
        if "down_proj" in k:
            # down_proj: [hidden_size, intermediate_size]
            num_n = int(torch.sum(v).item() / v.shape[0])
            total_neuron_budget += num_n

    if mode == "layer":
        print(f"[*] Generating RANDOM mask: Layer-wise matching mode.")
        # ... (Same as before)
        for k, v in original_mask.items():
            m = torch.zeros_like(v)
            if "down_proj" in k:
                num_n = int(torch.sum(v).item() / v.shape[0])
                indices = random.sample(range(v.shape[1]), num_n)
                m[:, indices] = 1.0
            else:
                num_n = int(torch.sum(v).item() / v.shape[1])
                indices = random.sample(range(v.shape[0]), num_n)
                m[indices, :] = 1.0
            random_mask[k] = m

    elif mode == "global":
        print(f"[*] Generating RANDOM mask: Global redistribution mode.")
        print(f"[*] Global budget: {total_neuron_budget} neurons total.")

        # 2. Build global pool
        all_possible_slots = []
        for l_idx in range(num_layers):
            for n_idx in range(intermediate_size):
                all_possible_slots.append((l_idx, n_idx))

        # 3. Sample
        if total_neuron_budget > len(all_possible_slots):
            total_neuron_budget = len(all_possible_slots)
        sampled_slots = random.sample(all_possible_slots, total_neuron_budget)

        # 4. Map back
        layer_assignments = {i: [] for i in range(num_layers)}
        for l_idx, n_idx in sampled_slots:
            layer_assignments[l_idx].append(n_idx)

        # 5. Construct tensors and print stats
        print(f"\n{'Layer':<8} | {'Random Neurons':<15} | {'Sparsity'}")
        print("-" * 40)

        total_random_masked_params = 0
        num_projs = len(target_projections)
        total_mlp_params = num_layers * (intermediate_size * hidden_size * num_projs)

        for l_idx in range(num_layers):
            indices = layer_assignments[l_idx]
            num_n = len(indices)

            # Layer statistics
            # In Gemma 2, MLP has 3 projections: up, gate, down
            num_projs = len(target_projections)
            layer_params_count = num_projs * (intermediate_size * hidden_size)
            masked_in_layer = num_projs * (num_n * hidden_size)
            total_random_masked_params += masked_in_layer

            layer_sparsity = (masked_in_layer / layer_params_count) * 100
            print(f"Layer {l_idx:<2} | {num_n:<14} | {layer_sparsity:.2f}%")

            # Apply to keys
            proj_keys = {
                f"model.layers.{l_idx}.mlp.up_proj.weight": "rows",
                f"model.layers.{l_idx}.mlp.gate_proj.weight": "rows",
                f"model.layers.{l_idx}.mlp.down_proj.weight": "cols"
            }

            for k, dim_type in proj_keys.items():
                if dim_type == "rows":
                    shape = (intermediate_size, hidden_size)
                    m = torch.zeros(shape)
                    if indices: m[indices, :] = 1.0
                else:
                    shape = (hidden_size, intermediate_size)
                    m = torch.zeros(shape)
                    if indices: m[:, indices] = 1.0
                random_mask[k] = m

        overall_sparsity = (total_random_masked_params / total_mlp_params) * 100
        print(f"\n" + "=" * 50)
        print(f"FINAL RANDOM GLOBAL MASK SUMMARY")
        print(f"[*] Total Masked Params: {int(total_random_masked_params):,}")
        print(f"[*] Overall Sparsity:    {overall_sparsity:.2f}%")
        print(f"=" * 50)

    return random_mask