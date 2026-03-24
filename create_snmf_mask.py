import torch
import json
from pathlib import Path
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
torch.serialization.add_safe_globals([GemmaTokenizerFast])

TARGET_CONCEPTS = ["multiplication_riddle", "division_riddle", "multiplication_symbolic", "division_symbolic"]


def generate_optimized_snmf_mask(
        model: torch.nn.Module,
        results_dir: str,
        scd_threshold: float = 0.3,
        purity_threshold: float = 0.4,
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
    affected_layers = 0

    for layer_folder in sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1])):
        layer_idx = int(layer_folder.name.split('_')[1])
        param_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"

        # We only count layers that actually exist in the model
        if not hasattr(model.model.layers[layer_idx].mlp, 'down_proj'):
            continue

        current_param = dict(model.named_parameters())[param_name]
        layer_total_params = current_param.numel()
        total_mlp_params += layer_total_params

        # Load data
        factors = torch.load(layer_folder / "snmf_factors.pt", map_location="cpu", weights_only=False)
        F = factors['F']
        with open(layer_folder / "feature_analysis_supervised.json", "r") as f:
            analysis = json.load(f)

        # Filter target latents
        target_latents = [int(idx) for idx, p in analysis.items()
                          if p['dominant_concept'] in target_concepts and
                          p['scd_score'] >= scd_threshold and
                          p['purity_score'] >= purity_threshold]

        if not target_latents:
            continue

        affected_layers += 1
        neuron_importance = F[:, target_latents].sum(dim=1)
        binary_neuron_mask = (neuron_importance > 0.1 * neuron_importance.max()).float()

        # Update Mask
        mask_2d = binary_neuron_mask.unsqueeze(0).expand_as(current_param.data)
        snmf_mask[param_name] = mask_2d

        # Stats for this layer
        layer_masked_count = mask_2d.sum().item()
        total_masked_params += layer_masked_count

    # --- Final Analysis Output ---
    remaining_params = total_mlp_params - total_masked_params
    noise_ratio = (total_masked_params / total_mlp_params) * 100 if total_mlp_params > 0 else 0

    print(f"\n" + "=" * 40)
    print(f"SNMF MASK GENERATION SUMMARY")
    print(f"=" * 40)
    print(f"[*] Layers with targeted features:  {affected_layers}")
    print(f"[*] Total MLP Down-Proj Params:     {int(total_mlp_params):,}")
    print(f"[*] Masked (Noised) Params:         {int(total_masked_params):,}")
    print(f"[*] Preserved (Clean) Params:       {int(remaining_params):,}")
    print(f"[*] Noise Ratio:                    {noise_ratio:.4f}%")
    print(f"=" * 40)

    return snmf_mask
