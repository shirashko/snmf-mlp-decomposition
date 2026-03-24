import argparse
import json
import os
import google.generativeai as gai
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import time
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
torch.serialization.add_safe_globals([GemmaTokenizerFast])

from utils import resolve_device, set_seed
from model_utils import load_local_model
from supervised_analysis import analyze_features_supervised
from unsupervised_analysis import analyze_features_unsupervised
from data_utils.concept_dataset import SupervisedConceptDataset

load_dotenv()


def extract_context_samples(latent_activations, token_ids, sample_ids, tokenizer, prompts, mass_threshold=0.9,
                            max_samples=15):
    sorted_idx = np.argsort(latent_activations)[::-1]
    sorted_acts = latent_activations[sorted_idx]

    cumulative_mass = np.cumsum(sorted_acts) / np.sum(sorted_acts)

    threshold_idx = np.searchsorted(cumulative_mass, mass_threshold)
    top_indices = sorted_idx[:min(threshold_idx + 1, max_samples)]

    contexts = []
    for idx in top_indices:
        s_id = sample_ids[idx]
        t_str = tokenizer.decode([token_ids[idx]])
        full_context = prompts[s_id] if s_id < len(prompts) else "Context missing"
        contexts.append(f"Token: '{t_str}' | Context: {full_context}")

    return contexts


class SimpleGeminiClient:
    def __init__(self, model_name: str = "models/gemini-2.5-flash", max_retries: int = 3, sleep_seconds: float = 5.0):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_TOKEN") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("No Gemini API key found. Please set GOOGLE_API_KEY or GEMINI_API_TOKEN.")
        gai.configure(api_key=api_key)
        self.model = gai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.model.generate_content(prompt)
                if not resp.candidates: raise RuntimeError("No candidates returned from Gemini.")
                text = getattr(resp, "text", None)
                if not isinstance(text, str) or not text.strip():
                    parts = getattr(getattr(resp.candidates[0], "content", None), "parts", None)
                    if parts: text = "\n".join(
                        [getattr(p, "text", "") for p in parts if getattr(p, "text", "").strip()])
                if not isinstance(text, str) or not text.strip(): raise RuntimeError("Gemini returned empty text.")
                return text.strip()
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()

                if "429" in err_msg or "quota" in err_msg:
                    wait_time = 60
                    print(
                        f"\n[Gemini] Rate limit hit (429). Sleeping for {wait_time}s before retry {attempt}/{self.max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[Gemini] error {e}, attempt {attempt}/{self.max_retries}")
                    if attempt < self.max_retries:
                        time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last_err}")


def generate_llm_label_and_test_cases(client, contexts):
    """
    Sends the top activating contexts to a language model to generate:
    1. A concise semantic label for the feature.
    2. 5 synthetic sentences that should activate this feature strongly.
    3. 5 neutral sentences that should not activate this feature.
    4. A boolean indicating if the feature is mathematical in nature.
    """
    prompt = f"""
    You are an expert in Mechanistic Interpretability. I will provide you with a list of contexts 
    where a specific latent feature in a language model activates strongly. 

    Contexts:
    {chr(10).join(contexts)}

    Based on these contexts:
    1. Provide a concise semantic LABEL for this feature.
    2. Generate 5 synthetic sentences that SHOULD activate this feature strongly.
    3. Generate 5 neutral sentences that SHOULD NOT activate this feature.
    4. Set "is_mathematical" to true if the feature relates to arithmetic, logic, or numbers.

    Return ONLY a JSON object:
    {{
        "label": "string",
        "is_mathematical": boolean,
        "activating_test_cases": ["str1", "str2", ...],
        "neutral_test_cases": ["str1", "str2", ...]
    }}
    """

    response_text = client.generate(prompt)
    try:
        clean_json = response_text.strip().strip('`').replace('json', '')
        return json.loads(clean_json)
    except:
        return {"error": "Failed to parse LLM response"}


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

    gemini_client = SimpleGeminiClient() if args.run_llm else None
    original_prompts = []

    if args.run_llm:
        if not args.data_path:
            raise ValueError("Must provide --data-path for LLM profiling.")
        dataset = SupervisedConceptDataset(args.data_path)
        original_prompts = [item[0] for item in dataset.get_data()]

    for layer_folder in sorted(results_dir.glob("layer_*")):
        layer_num = int(layer_folder.name.split("_")[1])
        factors_path = layer_folder / "snmf_factors.pt"

        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because it doesn't exist.")
            continue

        output_json_path = layer_folder / "feature_analysis_supervised.json"

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

        supervised_results = analyze_features_supervised(
            G, labels, sample_ids, token_ids, local_model.tokenizer,
            dominance_threshold=args.dominance_threshold, save_raw=args.save_raw, top_k=args.top_k_supervised
        )

        for k, v in existing_results.items():
            idx = int(k)
            if idx in supervised_results and 'llm_refined_profile' in v:
                supervised_results[idx]['llm_refined_profile'] = v['llm_refined_profile']

        if args.run_llm:
            print(f"Refining labels with Gemini for Layer {layer_num}...")
            for latent_idx in tqdm(range(G.shape[1]), desc="LLM Profiling"):

                if latent_idx in supervised_results and 'llm_refined_profile' in supervised_results[latent_idx]:
                    label = supervised_results[latent_idx]['llm_refined_profile'].get('label', '')
                    if label and "skipped" not in label.lower():
                        continue

                profile = supervised_results[latent_idx]
                is_math = any(kw in profile['dominant_concept'] for kw in ["riddle", "symbolic", "math"])
                is_pure = profile['purity_score'] > 0.7

                if is_math or is_pure:
                    contexts = extract_context_samples(G[:, latent_idx].numpy(), token_ids, sample_ids,
                                                       local_model.tokenizer, original_prompts, args.mass_threshold)

                    try:
                        llm_profile = generate_llm_label_and_test_cases(gemini_client, contexts)
                        supervised_results[latent_idx]['llm_refined_profile'] = llm_profile
                        time.sleep(6)
                    except Exception as e:
                        print(f"\n[Fatal] Stopping at feature {latent_idx} due to: {e}")
                        with open(output_json_path, 'w') as f:
                            json.dump(supervised_results, f, indent=2)
                        return

                else:
                    supervised_results[latent_idx]['llm_refined_profile'] = {"label": "skipped (low relevance)"}

                with open(output_json_path, 'w') as f:
                    json.dump(supervised_results, f, indent=2)

        # Unsupervised Analysis (Logit Lens)
        if not args.skip_vocab:
            unsupervised_results = analyze_features_unsupervised(F=F, local_model=local_model, layer=layer_num,
                                                                 mode=mode, top_k_tokens=args.top_k_unsupervised)
            with open(layer_folder / "feature_analysis_unsupervised.json", 'w') as f:
                json.dump(unsupervised_results, f, indent=2)


    print(f"\nAnalysis complete. Files saved in {args.results_dir}")

if __name__ == "__main__":
    main()


"""
python analyze_snmf_results.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --results-dir "./test_output" \
    --dominance-threshold 0.5 \
    --save-raw
"""