import torch
from tqdm import tqdm
from typing import List, Tuple, Dict
from contextlib import contextmanager
import sys


class LocalActivationGenerator:
    def __init__(self, local_model, data_device="cpu", mode="mlp"):
        self.local_model = local_model
        self.model = local_model.model
        self.tokenizer = local_model.tokenizer
        self.data_device = data_device
        self._mode = mode
        self.layer_modules = self.model.model.layers

    @contextmanager
    def _register_hooks(self, layers: List[int], storage: Dict):
        """Safely registers hooks and ensures they are removed after use."""
        hooks = []

        def get_hook_fn(layer_key):
            def hook_fn(module, input, output):
                target = input[0] if self._mode == 'mlp_intermediate' else output
                storage[layer_key] = target.detach().cpu()

            return hook_fn

        for idx, layer_num in enumerate(layers):
            module = self.layer_modules[layer_num].mlp
            # Note: Gemma-2 MLP structure may vary, ensuring we hook the correct projection
            target_module = module.down_proj if self._mode == 'mlp_intermediate' else module
            hooks.append(target_module.register_forward_hook(get_hook_fn(idx)))

        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    def generate_activations(
            self,
            prompts: List[str],
            layers: List[int],
            batch_size: int = 4,
            exclude_bos: bool = False
    ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        print(f"Generating activations for layers {layers}...")

        all_layer_acts = [[] for _ in layers]
        all_token_ids = []
        all_sample_ids = []
        mlp_storage = {}

        num_batches = (len(prompts) + batch_size - 1) // batch_size

        with torch.inference_mode():
            with self._register_hooks(layers, mlp_storage):
                for batch_idx in tqdm(range(num_batches), desc="Processing Batches", file=sys.stdout):
                    start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(prompts))

                    encoded = self.tokenizer(
                        prompts[start:end],
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(self.local_model.device)

                    # Forward pass
                    outputs = self.model(**encoded, output_hidden_states=(self._mode == 'residual'))

                    # Masking logic
                    mask = encoded["attention_mask"].bool().cpu()
                    input_ids_cpu = encoded["input_ids"].detach().cpu()

                    if exclude_bos and self.tokenizer.bos_token_id is not None:
                        mask &= (input_ids_cpu != self.tokenizer.bos_token_id)

                    # Collect metadata
                    for i, p_idx in enumerate(range(start, end)):
                        m = mask[i]
                        all_token_ids.extend(input_ids_cpu[i][m].tolist())
                        all_sample_ids.extend([p_idx] * m.sum().item())

                    # Extract activations from storage (already on CPU)
                    for idx, layer in enumerate(layers):
                        if self._mode == 'residual':
                            acts = outputs.hidden_states[layer + 1].detach().cpu()
                        else:
                            acts = mlp_storage[idx]  # Already CPU from hook

                        # Flatten and store
                        filtered_acts = acts[mask].view(-1, acts.size(-1))
                        all_layer_acts[idx].append(filtered_acts)

                    # --- Memory Management ---
                    del outputs
                    mlp_storage.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        final_acts = [torch.cat(layer_acts, dim=0) for layer_acts in all_layer_acts]
        return final_acts, all_token_ids, all_sample_ids