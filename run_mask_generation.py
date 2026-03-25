import torch
from transformers import AutoModelForCausalLM, AutoConfig
from create_snmf_mask import generate_optimized_snmf_mask


def run_main():
    # --- 1. Configurations ---
    model_path = "models/gemma2-2.03B_pretrained"
    results_dir = "./pretrained_results"
    output_mask_path = "snmf_pretrained_mask.pt"

    # Thresholds for surgical precision
    SCD_THRESHOLD = 0.2  # Minimal specificity
    PURITY_THRESHOLD = 0  # Minimal purity

    print(f"[*] Initializing model structure for {model_path}...")

    # We load only the config and create an empty model to save VRAM
    # This is enough to get parameter names and shapes for the mask
    config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # --- 2. Generate the Mask ---
    print(f"[*] Generating mask from SNMF results in: {results_dir}")

    # Note: If generate_optimized_snmf_mask is in the same file,
    # make sure it handles the 'meta' device by creating masks on CPU
    mask = generate_optimized_snmf_mask(
        model=model,
        results_dir=results_dir,
    )

    # --- 3. Save the Mask ---
    # Convert meta-device tensors to CPU before saving
    mask_to_save = {k: v for k, v in mask.items() if torch.any(v > 0)}

    print(f"[*] Saving mask with {len(mask_to_save)} affected layers to {output_mask_path}...")
    torch.save(mask_to_save, output_mask_path)

    print("\n[✔] Execution finished successfully.")
    print(f"You can now use '{output_mask_path}' in your distillation loop.")




if __name__ == "__main__":
    # Ensure the generate_optimized_snmf_mask function is defined above or imported
    run_main()