import torch
import json
from pathlib import Path
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast

# Add safety for loading
torch.serialization.add_safe_globals([GemmaTokenizerFast])

TARGET_CONCEPTS = ["multiplication_riddle", "division_riddle", "multiplication_symbolic", "division_symbolic"]


def generate_optimized_snmf_mask(
        model: torch.nn.Module,
        results_dir: str,
        scd_threshold: float = 0.25,
        purity_threshold: float = 0.3,
        target_concepts: list = None
):
    if target_concepts is None:
        target_concepts = TARGET_CONCEPTS

    results_path = Path(results_dir)
    snmf_mask = {}

    for name, param in model.named_parameters():
        snmf_mask[name] = torch.zeros(param.shape, device="cpu")

    total_mlp_params = 0
    total_masked_params = 0
    affected_layers_count = 0

    print(f"\n{'Layer':<10} | {'Masked (Up+Down)':<20} | {'Layer Noise %':<15} | {'Target Latents'}")
    print("-" * 75)

    # Sort layers numerically
    layer_folders = sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1]))

    for layer_folder in layer_folders:
        layer_idx = int(layer_folder.name.split('_')[1])

        # Define both components of the MLP for this layer
        down_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        up_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"

        # Check if model has these attributes
        if not hasattr(model.model.layers[layer_idx].mlp, 'down_proj'):
            continue

        # Get actual parameter objects to know the shapes
        params_dict = dict(model.named_parameters())
        down_param = params_dict[down_name]
        up_param = params_dict[up_name]

        layer_total_params = down_param.numel() + up_param.numel()
        total_mlp_params += layer_total_params

        # Load factors (F contains the neuron-to-latent mapping)
        factors = torch.load(layer_folder / "snmf_factors.pt", map_location="cpu", weights_only=False)
        F = factors['F']  # Shape: [Intermediate_Dim, Num_Latents]

        # Load the supervised analysis
        with open(layer_folder / "feature_analysis_supervised.json", "r") as f:
            analysis = json.load(f)

        # Filter latents based on updated, more inclusive thresholds
        target_latents = [
            int(idx) for idx, p in analysis.items()
            if p['dominant_concept'] in target_concepts and
               p['scd_score'] >= scd_threshold and
               p['purity_score'] >= purity_threshold
        ]

        if not target_latents:
            print(f"Layer {layer_idx:<3} | {0:<20} | {0.0:<14.2f}% | None")
            continue

        affected_layers_count += 1

        # 4. Identify critical neurons in the intermediate space
        # Sum importance across all target latents
        neuron_importance = F[:, target_latents].sum(dim=1)

        # Create a binary importance mask for neurons (Intermediate Dimension)
        # Using a 10% threshold relative to the max importance in this layer
        binary_neuron_mask = (neuron_importance > 0.1 * neuron_importance.max()).float()

        # --- MASK DOWN_PROJ ---
        # down_proj.weight shape: [Hidden_Dim, Intermediate_Dim]
        # binary_neuron_mask matches the Intermediate_Dim (columns of Down_Proj)
        mask_down = binary_neuron_mask.unsqueeze(0).expand_as(down_param.data)
        snmf_mask[down_name] = mask_down

        # --- MASK UP_PROJ ---
        # up_proj.weight shape: [Intermediate_Dim, Hidden_Dim]
        # binary_neuron_mask matches the Intermediate_Dim (rows of Up_Proj)
        mask_up = binary_neuron_mask.unsqueeze(1).expand_as(up_param.data)
        snmf_mask[up_name] = mask_up

        # Stats tracking
        layer_masked_count = int(mask_down.sum().item() + mask_up.sum().item())
        layer_noise_ratio = (layer_masked_count / layer_total_params) * 100
        total_masked_params += layer_masked_count

        print(
            f"Layer {layer_idx:<3} | {layer_masked_count:<20,} | {layer_noise_ratio:<14.2f}% | {len(target_latents)} latents")

    # --- Final Summary ---
    total_noise_ratio = (total_masked_params / total_mlp_params) * 100 if total_mlp_params > 0 else 0

    print("\n" + "=" * 50)
    print(f"OPTIMIZED SNMF MASK SUMMARY (Up + Down)")
    print(f"=" * 50)
    print(f"[*] Affected Layers:                {affected_layers_count}")
    print(f"[*] Total MLP Params Targeted:      {int(total_mlp_params):,}")
    print(f"[*] Masked (Noised) Params:         {int(total_masked_params):,}")
    print(f"[*] Overall Noise Ratio:            {total_noise_ratio:.4f}%")
    print(f"[*] Thresholds Used:                SCD={scd_threshold}, Purity={purity_threshold}")
    print(f"=" * 50)

    return snmf_mask