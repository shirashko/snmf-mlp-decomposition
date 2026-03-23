import torch
from typing import Dict, Any

from experiments.snmf_interp.generate_vocab_proj import get_vocab_proj_gemma_hf, get_vocab_proj_residual_hf
from model_utils import LocalModel



def analyze_features_unsupervised(
        F: torch.Tensor,
        local_model: LocalModel,
        layer: int,
        top_k_tokens: int = 30,
        mode: str = "mlp",
) -> Dict[int, Dict[str, Any]]:
    """
    Projects latent features into the vocabulary space using Logit Lens.

    This identifies the semantic 'direction' of each feature by observing which tokens
    it promotes or suppresses in the model's output distribution.

    Args:
        F (torch.Tensor): The feature matrix of shape (d, rank) from SNMF, where d hidden dimension
                          and rank is the number of latent features.
                          F[i, j] represents the contribution of feature j to the i-th hidden dimension.
        local_model (LocalModel): The loaded model containing the tokenizer and architecture.
        layer (int): The layer number corresponding to the features in F.
        top_k_tokens (int): Number of top tokens to retrieve for each feature projection.
        mode (str): Determines how to interpret the features. Options:
            - "mlp_intermediate": Projects the feature through the MLP down-projection and post-FFN layer norm.
            - "mlp": Projects the feature directly through the post-FFN layer norm.
            - "residual": Projects the raw feature vector as a residual without MLP transformations.
    """
    print(f"Analyzing features (unsupervised, layer {layer}, mode={mode})...")
    d_feat, rank = F.shape
    device = local_model.device
    hf_model = local_model.model
    tokenizer = local_model.tokenizer
    base_model = getattr(hf_model, "model", hf_model)
    layer_obj = base_model.layers[layer]

    feature_analysis = {}

    with torch.no_grad():
        for feature_idx in range(rank):
            feature_vec = F[:, feature_idx].to(device)

            # Transform feature based on the chosen mode
            if mode == "mlp_intermediate":
                # Project from d_mlp back to d_model
                concept_vector = layer_obj.mlp.down_proj(feature_vec.unsqueeze(0)).squeeze(0)
            elif mode == "mlp":
                concept_vector = feature_vec
            else:
                concept_vector = None  # Residual case

            # Apply LayerNorm if applicable (Gemma 2 specific)
            if concept_vector is not None:
                ln_layer = getattr(layer_obj, "post_feedforward_layernorm", None)
                if ln_layer:
                    concept_vector = ln_layer(concept_vector.unsqueeze(0)).squeeze(0)

            # Vocabulary Projection
            if mode in ["mlp_intermediate", "mlp"]:
                pos_vals, pos_idx = get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k_tokens, device)
                neg_vals, neg_idx = get_vocab_proj_gemma_hf(-concept_vector, hf_model, top_k_tokens, device)
            else:
                pos_vals, pos_idx = get_vocab_proj_residual_hf(feature_vec, hf_model, top_k_tokens, device)
                neg_vals, neg_idx = get_vocab_proj_residual_hf(-feature_vec, hf_model, top_k_tokens, device)

            # Combining both positive and negative for a single batch call
            all_indices = torch.cat([pos_idx, neg_idx]).tolist()
            all_tokens = tokenizer.batch_decode([[i] for i in all_indices])

            pos_tokens = all_tokens[:top_k_tokens]
            neg_tokens = all_tokens[top_k_tokens:]

            feature_analysis[feature_idx] = {
                'positive_tokens': pos_tokens,
                'positive_logits': pos_vals.cpu().tolist(),
                'negative_tokens': neg_tokens,
                'negative_logits': (-neg_vals).cpu().tolist(),
                'interpretation': f"Top promotes: {', '.join(pos_tokens[:5])}",
            }

    return feature_analysis