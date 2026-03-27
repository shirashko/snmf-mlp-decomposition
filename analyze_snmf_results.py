import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import time
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
torch.serialization.add_safe_globals([GemmaTokenizerFast])

from utils import resolve_device, set_seed
from model_utils import load_local_model
from supervised_analysis import analyze_features_supervised, plot_layer_concept_trends
from unsupervised_analysis import analyze_features_unsupervised
from data_utils.concept_dataset import SupervisedConceptDataset
from feature_interpreter import FeatureInterpreter

load_dotenv()



def main():
    parser = argparse.ArgumentParser(description="Analyze pre-trained SNMF results.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True, help="Path to folder containing layer_X subfolders")
    parser.add_argument("--dominance-threshold", type=float, default=0.5)
    parser.add_argument("--skip-vocab", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--mass-threshold", type=float, default=0.9)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--top-k-unsupervised", type=int, default=30)
    parser.add_argument("--top-k-supervised", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    interpreter = None
    if args.run_llm:
        if not args.data_path:
            raise ValueError("Must provide --data-path for LLM profiling.")

        dataset = SupervisedConceptDataset(args.data_path)
        original_prompts = [item[0] for item in dataset.get_data()]

        interpreter = FeatureInterpreter(
            tokenizer=local_model.tokenizer,
            prompts=original_prompts,
            mass_threshold=args.mass_threshold
        )

    for layer_folder in sorted(results_dir.glob("layer_*")):
        layer_num = int(layer_folder.name.split("_")[1])
        factors_path = layer_folder / "snmf_factors.pt"

        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because it doesn't exist.")
            continue

        output_json_path = layer_folder / "feature_analysis_supervised.json"

        # Load existing results for Resume logic
        existing_results = {}
        if output_json_path.exists():
            try:
                with open(output_json_path, 'r') as f:
                    existing_results = json.load(f)
            except Exception as e:
                print(f"Warning: could not load existing JSON: {e}")

        print(f"\nProcessing {layer_folder.name}...")
        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)
        F, G = checkpoint['F'], checkpoint['G']
        token_ids, sample_ids = checkpoint['token_ids'], checkpoint['sample_ids']
        labels, mode = checkpoint['labels'], checkpoint.get('mode', 'mlp_intermediate')

        # 1. Base statistical profiling
        supervised_results = analyze_features_supervised(
            G, labels, sample_ids, token_ids, local_model.tokenizer,
            dominance_threshold=args.dominance_threshold, save_raw=args.save_raw, top_k=args.top_k_supervised
        )

        # 2. Merge existing LLM profiles back to supervised_results
        # (This prevents re-running indices that already have an LLM label)
        for k, v in existing_results.items():
            idx = int(k)
            if idx in supervised_results and 'llm_refined_profile' in v:
                supervised_results[idx]['llm_refined_profile'] = v['llm_refined_profile']

        with open(output_json_path, 'w') as f:
            json.dump(supervised_results, f, indent=2)
        # 3. LLM Semantic Refinement
        if args.run_llm:
            print(f"Refining labels with Gemini for Layer {layer_num}...")
            for latent_idx in tqdm(range(G.shape[1]), desc="LLM Profiling"):

                # Resume Check: Skip if already interpreted by LLM
                if 'llm_refined_profile' in supervised_results[latent_idx]:
                    label = supervised_results[latent_idx]['llm_refined_profile'].get('label', '')
                    if label and "skipped" not in label.lower():
                        continue

                profile = supervised_results[latent_idx]
                # Filter for relevant features (Math keywords or High Purity)
                is_math = any(
                    kw in profile['dominant_concept'].lower() for kw in ["riddle", "symbolic"])
                is_pure = profile['purity_score'] > 0.7

                if is_math or is_pure:
                    try:
                        # Use the FeatureInterpreter class
                        llm_profile = interpreter.explain_feature(
                            G[:, latent_idx].numpy(),
                            token_ids,
                            sample_ids
                        )
                        supervised_results[latent_idx]['llm_refined_profile'] = llm_profile

                        # Delay to respect 15 RPM Rate Limit (Free Tier)
                        time.sleep(6)
                    except Exception as e:
                        print(f"\n[Fatal] Stopping at feature {latent_idx} due to error: {e}")
                        with open(output_json_path, 'w') as f:
                            json.dump(supervised_results, f, indent=2)
                        return  # Exit main to allow for manual intervention or wait
                else:
                    supervised_results[latent_idx]['llm_refined_profile'] = {"label": "skipped (low relevance)"}

                # Save after every successful feature to ensure progress is persistent
                with open(output_json_path, 'w') as f:
                    json.dump(supervised_results, f, indent=2)

        # Unsupervised Vocabulary Projection (Logit Lens)
        if not args.skip_vocab:
            unsupervised_results = analyze_features_unsupervised(
                F=F,
                local_model=local_model,
                layer=layer_num,
                mode=mode,
                top_k_tokens=args.top_k_unsupervised
            )
            with open(layer_folder / "feature_analysis_unsupervised.json", 'w') as f:
                json.dump(unsupervised_results, f, indent=2)

    print("\nGenerating model-wide trend plots...")
    try:
        plot_layer_concept_trends(args.results_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print(f"\nAnalysis complete. Files saved in {args.results_dir}")


if __name__ == "__main__":
    main()


"""
python analyze_snmf_results.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --results-dir "./final_run_all_layers" \
    --data-path "data/data_subsampled.json" \
    --dominance-threshold 0.4 \
    --top-k-supervised 50 \
    --top-k-unsupervised 64 \
    --save-raw
    
    
python analyze_snmf_results.py \
    --model-path "models/gemma2-2.03B_pretrained" \
    --results-dir "./pretrained_results" \
    --data-path "data/data_subsampled.json" \
    --dominance-threshold 0.4 \
    --top-k-supervised 50 \
    --top-k-unsupervised 64 \
    --save-raw
"""