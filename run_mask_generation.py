import torch
from transformers import AutoModelForCausalLM, AutoConfig
from create_snmf_mask import generate_optimized_snmf_mask, generate_random_matching_mask


def run_main():
    model_path = "models/gemma2-2.03B_pretrained"
    results_dir = "./pretrained_results"
    shared_name_prefix = "snmf_mask"
    output_mask_path = f"{shared_name_prefix}.pt"
    output_random_mask_path = f"random_baseline_{shared_name_prefix}.pt"

    print(f"[*] Initializing model structure for {model_path}...")

    # Load config and initialize empty model on meta device to save memory
    config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # 1. Generate Localized SNMF Mask
    snmf_mask = generate_optimized_snmf_mask(
        model=model,
        results_dir=results_dir,
        threshold=0.02,
        purity_threshold=0.35,
        target_projections=["down_proj"]
    )

    # 2. Generate Random Baseline Mask for comparison
    random_mask = generate_random_matching_mask(snmf_mask, config, mode="global", target_projections=["down_proj"])

    # 3. Filter and Save Masks
    snmf_to_save = {k: v for k, v in snmf_mask.items() if torch.any(v > 0)}
    random_to_save = {k: v for k, v in random_mask.items() if torch.any(v > 0)}

    print(f"[*] Saving SNMF mask to {output_mask_path}...")
    torch.save(snmf_to_save, output_mask_path)

    print(f"[*] Saving Random baseline mask to {output_random_mask_path}...")
    torch.save(random_to_save, output_random_mask_path)

    print("\n[✔] Execution finished successfully.")
    print(f"You can now use both masks to compare localization vs random noise in your evaluation.")


if __name__ == "__main__":
    run_main()