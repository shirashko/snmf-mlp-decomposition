import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
import pickle
from datetime import datetime

import torch

# TransformerLens imports - optional (only needed for TransformerLens-based models)
try:
    from transformer_lens import HookedTransformer
    from transformer_lens.utilities.addmm import batch_addmm  # used for Gemma path

    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    HookedTransformer = None
    batch_addmm = None
    TRANSFORMER_LENS_AVAILABLE = False

# These imports are also optional for standalone HF usage
try:
    from evaluation.json_handler import JsonHandler
    from intervention.intervener import Intervener
    from factorization.seminmf import NMFSemiNMF
except ImportError:
    JsonHandler = None
    Intervener = None
    NMFSemiNMF = None


# ------------------------------
# Utils
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(txt: str):
    print(f"[{datetime.now()}] {txt}", flush=True)


def parse_int_list(spec: str):
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    if spec is None or spec.strip() == "":
        return out
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(part))
    return out


# ------------------------------
# Model-specific helpers (TransformerLens)
# ------------------------------
@torch.no_grad()
def get_vocab_proj_regular(A, model, layer, top_k=50, device="cuda"):
    """For TransformerLens models (non-Gemma)."""
    basis_units = model.W_out
    direction = model.ln_final(A @ basis_units[layer])
    vocab_proj = model.unembed(direction.to(device))
    values, indices = torch.topk(vocab_proj, top_k)
    return values, indices


@torch.no_grad()
def get_concept_vector_gemma(mlp_vec, model, layer, device="cuda"):
    """For TransformerLens Gemma models."""
    return model.blocks[layer].ln2_post(
        batch_addmm(model.b_out[layer], model.W_out[layer], mlp_vec.to(device))
    )


@torch.no_grad()
def get_vocab_proj_gemma(concept_vector, model, top_k=50, device="cuda"):
    """For TransformerLens Gemma models."""
    direction = model.ln_final(concept_vector)
    vocab_proj = model.unembed(direction.to(device))
    values, indices = torch.topk(vocab_proj, top_k)
    return values, indices


# ------------------------------
# HuggingFace model helpers (for local/custom models)
# ------------------------------
@torch.no_grad()
def get_concept_vector_gemma_hf(mlp_vec, hf_model, layer, device="cuda"):
    """
    For HuggingFace Gemma-2 models (e.g., local reduced models).

    Equivalent to TransformerLens version:
        model.blocks[layer].ln2_post(batch_addmm(model.b_out[layer], model.W_out[layer], mlp_vec))

    Args:
        mlp_vec: Feature vector in MLP intermediate space (d_mlp,)
        hf_model: HuggingFace AutoModelForCausalLM
        layer: Layer index
        device: Torch device

    Returns:
        concept_vector in residual stream space (d_model,)
    """
    # Get the MLP's down_proj (equivalent to W_out in TransformerLens)
    # down_proj: Linear(intermediate_size -> hidden_size)
    down_proj = hf_model.model.layers[layer].mlp.down_proj

    # Apply down_proj: output = mlp_vec @ weight.T + bias
    residual_vec = down_proj(mlp_vec.unsqueeze(0).to(device)).squeeze(0)  # (hidden_size,)

    # Apply post-feedforward layer norm (Gemma-2 specific, equivalent to ln2_post)
    post_ff_ln = hf_model.model.layers[layer].post_feedforward_layernorm
    concept_vector = post_ff_ln(residual_vec.unsqueeze(0)).squeeze(0)  # (hidden_size,)

    return concept_vector


@torch.no_grad()
def get_vocab_proj_gemma_hf(concept_vector, hf_model, top_k=50, device="cuda"):
    """
    For HuggingFace Gemma-2 models.

    Equivalent to TransformerLens version:
        direction = model.ln_final(concept_vector)
        vocab_proj = model.unembed(direction)

    Args:
        concept_vector: Vector in residual stream space (d_model,)
        hf_model: HuggingFace AutoModelForCausalLM
        top_k: Number of top tokens to return
        device: Torch device

    Returns:
        (values, indices) - top-k logit values and token indices
    """
    # Apply final layer norm (equivalent to ln_final)
    direction = hf_model.model.norm(concept_vector.unsqueeze(0).to(device))  # (1, hidden_size)

    # Apply unembedding via lm_head (equivalent to unembed)
    vocab_proj = hf_model.lm_head(direction).squeeze()  # (vocab_size,)

    # Get top-k
    values, indices = torch.topk(vocab_proj, top_k)
    return values, indices


@torch.no_grad()
def get_vocab_proj_residual_hf(residual_vec, hf_model, top_k=50, device="cuda"):
    """
    For HuggingFace models when feature is already in residual stream space.
    Just applies final norm and unembeds.

    Args:
        residual_vec: Vector in residual stream space (d_model,)
        hf_model: HuggingFace AutoModelForCausalLM
        top_k: Number of top tokens to return
        device: Torch device

    Returns:
        (values, indices) - top-k logit values and token indices
    """
    direction = hf_model.model.norm(residual_vec.unsqueeze(0).to(device))
    vocab_proj = hf_model.lm_head(direction).squeeze()
    values, indices = torch.topk(vocab_proj, top_k)
    return values, indices


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vocab projection for NMF concept vectors (auto-switch Gemma vs Regular)."
    )
    # Core model/config
    parser.add_argument("--model-name", type=str, required=True,
                        help='e.g. "gemma-2-2b" or "meta-llama/Llama-3.1-8B"')
    parser.add_argument("--device", type=str, default=None,
                        help='torch device, e.g. "cuda", "cpu", or "mps". Default: auto')
    parser.add_argument("--seed", type=int, default=42)

    # Data layout (auto layout uses base-path unless you override below)
    parser.add_argument("--base-path", type=str, required=True,
                        help="Project root used for default factorization/output paths.")

    parser.add_argument("--factorization-base-path", type=str, default=None, required=True,
                        help="Override the directory that contains NMF pickle files per rank.")
    parser.add_argument("--output-path", type=str, default=None, required=True,
                        help="Override the output JSON path.")

    # Workload
    parser.add_argument("--ranks", type=str, default="100",
                        help="Ranks to process, e.g. '100' or '64,128' (list).")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layers to process (range/list). Examples: '0-25' or '0,2,5-7'. "
                             "Default depends on model: 26 layers for Gemma-2-2B, 32 for Llama 8B.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k vocab tokens to save per concept (+ and -).")

    # Metadata
    parser.add_argument("--sparsity", type=float, default=0.01,
                        help="Logged in output. Gemma path also uses it in filename tag 's{value}'.")

    args = parser.parse_args()

    # Repro
    set_seed(args.seed)

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    log(f"Job started. device={device}")
    is_gemma = "gemma" in args.model_name.lower()

    # Layers
    if args.layers is None:
        default_layers = list(range(26)) if is_gemma else list(range(32))
        layers = default_layers
    else:
        layers = parse_int_list(args.layers)

    # Ranks
    ranks = [int(x) for x in args.ranks.split(",")]

    # Paths (auto layout unless overridden)
    factorization_base_path = args.factorization_base_path
    save_path = args.output_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Init model & intervener
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    _ = Intervener(model)  # keep parity with your originals

    # JSON writer
    json_handler = JsonHandler(
        ["K", "layer", "h_row", "top_logit_values", "top_shifted_tokens", "intervention_sign", "sparsity"],
        save_path,
        auto_write=False
    )

    # Process
    for rank in ranks:
        log(f"Starting rank {rank}...")
        # Load all NMF models for this rank
        nmf_models = {}
        for layer in layers:
            fn = f"nmf-l{layer}-r{rank}.pkl"
            rank_dir = os.path.join(factorization_base_path, str(layer), str(rank))
            if not os.path.isdir(rank_dir):
                log(f"Dir does not exist, skipping rank: {rank}  ({rank_dir})")
                continue
            fp = os.path.join(rank_dir, fn)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    nmf_models[layer] = pickle.load(f)
                log(f"Loaded NMF model for layer {layer}, rank {rank}")
            else:
                log(f"Missing NMF file for layer {layer}, rank {rank} → skipping")

        # Vocab projection per layer/h
        for layer in layers:
            nmf: NMFSemiNMF = nmf_models.get(layer)
            if nmf is None:
                continue

            for h_idx in range(rank):
                with torch.no_grad():
                    if is_gemma:
                        mlp_vec = nmf.F_.T[h_idx].to(device)
                        concept_vector = get_concept_vector_gemma(mlp_vec, model, layer, device=device)

                        pos_vals_t, pos_idx_t = get_vocab_proj_gemma(concept_vector, model, top_k=args.top_k,
                                                                     device=device)
                        neg_vals_t, neg_idx_t = get_vocab_proj_gemma(-concept_vector, model, top_k=args.top_k,
                                                                     device=device)

                        sparsity_meta = f"s{args.sparsity}"
                    else:
                        concept_vector = (nmf.F_.T[h_idx] / nmf.F_.T[h_idx].norm()).to(device)

                        pos_vals_t, pos_idx_t = get_vocab_proj_regular(concept_vector, model, layer, top_k=args.top_k,
                                                                       device=device)
                        neg_vals_t, neg_idx_t = get_vocab_proj_regular(-concept_vector, model, layer, top_k=args.top_k,
                                                                       device=device)

                        sparsity_meta = args.sparsity

                    pos_vals = pos_vals_t.tolist()
                    pos_toks = [model.to_str_tokens([tid]) for tid in pos_idx_t]
                    neg_vals = neg_vals_t.tolist()
                    neg_toks = [model.to_str_tokens([tid]) for tid in neg_idx_t]

                    json_handler.add_row(
                        K=rank,
                        layer=layer,
                        h_row=h_idx,
                        top_logit_values=pos_vals,
                        top_shifted_tokens=pos_toks,
                        intervention_sign=1,
                        sparsity=sparsity_meta
                    )
                    json_handler.add_row(
                        K=rank,
                        layer=layer,
                        h_row=h_idx,
                        top_logit_values=neg_vals,
                        top_shifted_tokens=neg_toks,
                        intervention_sign=-1,
                        sparsity=sparsity_meta
                    )
                    del concept_vector

            log(f"Finished Processing layer: {layer}, rank: {rank}")

        del nmf_models
        json_handler.write()

    log(f"Wrote: {save_path}")
    log("Job finished.")
