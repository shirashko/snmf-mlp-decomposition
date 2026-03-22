import torch
from tqdm import tqdm
from typing import List, Tuple
from model_utils import LocalModel


class LocalActivationGenerator:
    """
    Activation generator for local models using HuggingFace.

    Supports:
    - 'mlp': MLP output activations (what SNMF paper uses)
    - 'residual': Residual stream after each layer
    """

    def __init__(
            self,
            local_model: LocalModel,
            data_device: str = "cpu",
            mode: str = "mlp"
    ):
        self.local_model = local_model
        self.model = local_model.model
        self.tokenizer = local_model.tokenizer
        self.data_device = data_device
        self._mode = mode

        if mode not in ['mlp', 'mlp_intermediate', 'residual']:
            raise ValueError(f"Mode '{mode}' not supported. Use 'mlp', 'mlp_intermediate', or 'residual'.")

    def _get_mlp_module(self, layer: int):
        """Get the MLP module for a given layer (Gemma-2 specific)."""
        # Gemma-2 structure: model.layers[layer].mlp
        return self.model.model.layers[layer].mlp

    def _get_down_proj(self, layer: int):
        """Get the down_proj layer for capturing intermediate activations (Gemma-2 specific)."""
        # Gemma-2 structure: model.layers[layer].mlp.down_proj
        return self.model.model.layers[layer].mlp.down_proj

    def generate_activations(
            self,
            prompts: List[str],
            layers: List[int],
            batch_size: int = 4,
    ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """
        Generate activations for multiple layers using HuggingFace.

        Returns:
            (activations_per_layer, token_ids, sample_ids_per_token)
        """
        print(f"Generating activations for layers {layers}...")
        print(f"  Mode: {self._mode}")

        all_layer_acts = [[] for _ in layers]
        all_token_ids = []
        all_sample_ids = []

        pad_token_id = self.tokenizer.pad_token_id

        num_batches = (len(prompts) + batch_size - 1) // batch_size

        # For MLP modes, we need to register hooks
        if self._mode in ['mlp', 'mlp_intermediate']:
            # Storage for hook outputs
            mlp_outputs = {}
            hooks = []

            if self._mode == 'mlp':
                # Hook on MLP output (post-down_proj, 320 dim)
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        mlp_outputs[layer_idx] = output.detach()

                    return hook_fn

                for layer_idx, layer in enumerate(layers):
                    mlp_module = self._get_mlp_module(layer)
                    hook = mlp_module.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)

            else:  # mlp_intermediate
                # Hook on down_proj INPUT to get intermediate activations (1280 dim)
                # This is what the original SNMF repo uses
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        # input is a tuple, first element is the intermediate activations
                        mlp_outputs[layer_idx] = input[0].detach()

                    return hook_fn

                for layer_idx, layer in enumerate(layers):
                    down_proj = self._get_down_proj(layer)
                    hook = down_proj.register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)

        try:
            with torch.no_grad():
                for batch_idx in tqdm(range(num_batches), desc="Collecting activations"):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(prompts))
                    batch_prompts = prompts[start:end]

                    # Tokenize with left padding
                    encoded = self.tokenizer(
                        batch_prompts,
                        padding=True,
                        padding_side="left",
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = encoded["input_ids"].to(self.local_model.device)
                    attention_mask = encoded["attention_mask"].to(self.local_model.device)

                    # Clear MLP outputs from previous batch
                    if self._mode in ['mlp', 'mlp_intermediate']:
                        mlp_outputs.clear()

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=(self._mode == 'residual'),
                    )

                    # Mask for non-padding tokens
                    mask = attention_mask.bool()

                    # Get token IDs for this batch
                    for i, prompt_idx in enumerate(range(start, end)):
                        prompt_mask = mask[i]
                        prompt_tokens = input_ids[i][prompt_mask]
                        all_token_ids.extend(prompt_tokens.cpu().tolist())
                        all_sample_ids.extend([prompt_idx] * prompt_mask.sum().item())

                    # Extract activations per layer
                    if self._mode in ['mlp', 'mlp_intermediate']:
                        # Use hook outputs
                        # mlp: (batch, seq, d_model=320) - post-down_proj
                        # mlp_intermediate: (batch, seq, d_mlp=1280) - pre-down_proj
                        for layer_idx, layer in enumerate(layers):
                            if layer_idx in mlp_outputs:
                                acts = mlp_outputs[layer_idx]
                                nonpad_acts = acts[mask].view(-1, acts.size(-1))
                                all_layer_acts[layer_idx].append(nonpad_acts.cpu())
                            else:
                                print(f"  Warning: No MLP output captured for layer {layer}")
                    else:
                        # Use hidden_states for residual mode
                        hidden_states = outputs.hidden_states
                        for layer_idx, layer in enumerate(layers):
                            if layer + 1 < len(hidden_states):
                                acts = hidden_states[layer + 1].detach()
                                nonpad_acts = acts[mask].view(-1, acts.size(-1))
                                all_layer_acts[layer_idx].append(nonpad_acts.cpu())
                            else:
                                print(f"  Warning: Layer {layer} not available")

                    # Clear memory
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            # Remove hooks
            if self._mode in ['mlp', 'mlp_intermediate']:
                for hook in hooks:
                    hook.remove()

        # Concatenate
        final_acts = [torch.cat(layer_acts, dim=0) if layer_acts else torch.empty(0)
                      for layer_acts in all_layer_acts]

        for i, layer in enumerate(layers):
            print(f"  Layer {layer}: {final_acts[i].shape}")

        return final_acts, all_token_ids, all_sample_ids
