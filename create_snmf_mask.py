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
    affected_layers_count = 0

    print(f"\n{'Layer':<10} | {'Masked Params':<15} | {'Layer Noise %':<15} | {'Target Latents'}")
    print("-" * 65)

    for layer_folder in sorted(results_path.glob("layer_*"), key=lambda x: int(x.name.split('_')[1])):
        layer_idx = int(layer_folder.name.split('_')[1])
        param_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"

        if not hasattr(model.model.layers[layer_idx].mlp, 'down_proj'):
            continue

        current_param = dict(model.named_parameters())[param_name]
        layer_total_params = current_param.numel()
        total_mlp_params += layer_total_params

        # Load factors and analysis
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
            print(f"Layer {layer_idx:<3} | {0:<15} | {0.0:<14.2f}% | None")
            continue

        affected_layers_count += 1
        neuron_importance = F[:, target_latents].sum(dim=1)

        # Binary mask creation
        binary_neuron_mask = (neuron_importance > 0.1 * neuron_importance.max()).float()
        mask_2d = binary_neuron_mask.unsqueeze(0).expand_as(current_param.data)
        snmf_mask[param_name] = mask_2d

        # Layer stats
        layer_masked_count = int(mask_2d.sum().item())
        layer_noise_ratio = (layer_masked_count / layer_total_params) * 100

        total_masked_params += layer_masked_count

        print(
            f"Layer {layer_idx:<3} | {layer_masked_count:<15,} | {layer_noise_ratio:<14.2f}% | {len(target_latents)} latents")

    # --- Final Analysis Output ---
    remaining_params = total_mlp_params - total_masked_params
    total_noise_ratio = (total_masked_params / total_mlp_params) * 100 if total_mlp_params > 0 else 0

    print("\n" + "=" * 40)
    print(f"SNMF MASK GENERATION SUMMARY")
    print(f"=" * 40)
    print(f"[*] Affected Layers:                {affected_layers_count}")
    print(f"[*] Total MLP Down-Proj Params:     {int(total_mlp_params):,}")
    print(f"[*] Masked (Noised) Params:         {int(total_masked_params):,}")
    print(f"[*] Preserved (Clean) Params:       {int(remaining_params):,}")
    print(f"[*] Overall Noise Ratio:            {total_noise_ratio:.4f}%")
    print(f"=" * 40)

    return snmf_mask