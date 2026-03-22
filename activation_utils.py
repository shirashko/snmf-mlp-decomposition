import torch
from tqdm import tqdm
from typing import List


class LocalActivationGenerator:
    def __init__(self, local_model, data_device="cpu", mode="mlp"):
        self.local_model = local_model
        self.data_device = data_device
        self._mode = mode

    def _get_hooks(self, layers, mlp_outputs):
        hooks = []
        for i, layer in enumerate(layers):
            # Target the specific MLP layer for Gemma-2
            target = self.local_model.model.model.layers[layer].mlp
            if self._mode == 'mlp_intermediate':
                target = target.down_proj

            def hook_fn(l_idx):
                return lambda m, i, o: mlp_outputs.update(
                    {l_idx: (i[0] if self._mode == 'mlp_intermediate' else o).detach()})

            hooks.append(target.register_forward_hook(hook_fn(i)))
        return hooks

    def generate_activations(self, prompts: List[str], layers: List[int], batch_size: int = 4):
        all_layer_acts = [[] for _ in layers]
        all_token_ids, all_sample_ids = [], []
        mlp_outputs = {}

        hooks = self._get_hooks(layers, mlp_outputs) if self._mode != 'residual' else []

        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(prompts), batch_size)):
                    batch = prompts[i: i + batch_size]
                    enc = self.local_model.tokenizer(batch, padding=True, padding_side="left", return_tensors="pt").to(
                        self.local_model.device)
                    out = self.local_model.model(**enc, output_hidden_states=(self._mode == 'residual'))

                    mask = enc.attention_mask.bool()
                    for b in range(len(batch)):
                        all_token_ids.extend(enc.input_ids[b][mask[b]].cpu().tolist())
                        all_sample_ids.extend([i + b] * mask[b].sum().item())

                    for l_idx, layer in enumerate(layers):
                        acts = out.hidden_states[layer + 1] if self._mode == 'residual' else mlp_outputs[l_idx]
                        all_layer_acts[l_idx].append(acts[mask].cpu())
        finally:
            for h in hooks: h.remove()

        return [torch.cat(l, dim=0) for l in all_layer_acts], all_token_ids, all_sample_ids