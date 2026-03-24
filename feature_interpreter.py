import json
from typing import List, Dict, Any
import numpy as np
import os
import time
import google.generativeai as gai
from typing import Optional


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


class FeatureInterpreter:
    def __init__(self, tokenizer, prompts: List[str], mass_threshold: float = 0.9):
        self.client = SimpleGeminiClient()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.mass_threshold = mass_threshold

    def extract_contexts(self, latent_activations: np.ndarray, token_ids: List[int], sample_ids: List[int],
                         max_samples: int = 15):
        """Identifies tokens composing the top 90% mass of activation to provide clean context."""
        sorted_idx = np.argsort(latent_activations)[::-1]
        sorted_acts = latent_activations[sorted_idx]

        cumulative_mass = np.cumsum(sorted_acts) / (np.sum(sorted_acts) + 1e-9)
        threshold_idx = np.searchsorted(cumulative_mass, self.mass_threshold)
        top_indices = sorted_idx[:min(threshold_idx + 1, max_samples)]

        contexts = []
        for idx in top_indices:
            s_id = sample_ids[idx]
            t_str = self.tokenizer.decode([token_ids[idx]])
            full_context = self.prompts[s_id] if s_id < len(self.prompts) else "Context missing"
            contexts.append(f"Token: '{t_str}' | Context: {full_context}")
        return contexts

    def explain_feature(self, latent_activations: np.ndarray, token_ids: List[int], sample_ids: List[int]) -> Dict[
        str, Any]:
        """Sends extracted contexts to the LLM and returns a structured interpretation."""
        contexts = self.extract_contexts(latent_activations, token_ids, sample_ids)

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

        response_text = self.client.generate(prompt)
        try:
            clean_json = response_text.strip().strip('`').replace('json', '')
            return json.loads(clean_json)
        except Exception as e:
            return {"error": f"Failed to parse LLM response: {str(e)}", "raw": response_text}