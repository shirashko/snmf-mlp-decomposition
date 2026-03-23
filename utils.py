import os
import re
import pickle
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Basic Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str) -> str:
    spec = spec.lower()
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _safe_concept(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]+", "_", name.strip().replace(" ", "_"))).strip("_")


def _safe_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if t is None: continue
        if not isinstance(t, str): t = str(t)
        out.append(t.replace("\r", "\\r").replace("\n", "\\n"))
    return out


# ---------------------------------------------------------------------------
# Path & IO Management
# ---------------------------------------------------------------------------

def get_pipeline_path(
        outdir: str, model_safe: str, file_type: str, rank: int, n_samples: int,
        concept_safe: str, track: str, filename: str
) -> str:
    """
    Constructs paths matching:
    outdir -> model -> pickles/csvs -> rank&sentences -> concept -> mlp/embeddings -> filename
    """
    from pathlib import Path
    d = Path(outdir) / model_safe / file_type / f"rank{rank}_{n_samples}sentences" / concept_safe / track
    d.mkdir(parents=True, exist_ok=True)
    return str(d / filename)


def save_df_to_csv(df: pd.DataFrame, path: str, dedupe_cols: List[str] = None) -> None:
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df], ignore_index=True)

    if not df.empty and dedupe_cols and set(dedupe_cols).issubset(df.columns):
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
        df = df.sort_values(dedupe_cols, ascending=[True] * len(dedupe_cols))

    df.to_csv(path, index=False)


def load_potential_feature_keys(
        root_dir: str, potential_root: str, model_short: str, concept_safe: str, rank: int, n_samples: int
) -> set:
    from pathlib import Path
    pot_path = Path(
        root_dir) / potential_root / model_short / f"{concept_safe}_rank{rank}_{n_samples}sentences" / "potential_features.csv"
    if not pot_path.exists():
        return set()
    try:
        dfp = pd.read_csv(pot_path)
        if {"layer", "feature"}.issubset(dfp.columns):
            return set(zip(dfp["layer"].astype(int), dfp["feature"].astype(int)))
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Model / Activation Extraction
# ---------------------------------------------------------------------------


def get_special_token_ids(model) -> set:
    tok = model.tokenizer
    return {int(getattr(tok, attr)) for attr in ["pad_token_id", "bos_token_id", "eos_token_id"] if
            getattr(tok, attr, None) is not None}


def vector_to_logits(model: Any, v: Tensor, use_ln_final: bool = True) -> Tensor:
    v = v.to(device=model.W_U.device, dtype=model.W_U.dtype)
    if use_ln_final and hasattr(model, "ln_final") and model.ln_final is not None:
        v = model.ln_final(v)
    return model.unembed(v)


def generate_token_contexts(tokens: Sequence[int], sample_ids: Sequence[int], act_generator: Any,
                            context_window: int = 15) -> List[Tuple[str, str]]:
    assert len(tokens) == len(sample_ids)
    token_ds = []
    for i in range(len(tokens)):
        current_sample_id = sample_ids[i]
        token_str = act_generator.model.to_str_tokens([tokens[i]])[0][0]
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)
        context_tokens = [act_generator.model.to_str_tokens([tokens[j]])[0][0] for j in range(start, end) if
                          sample_ids[j] == current_sample_id]
        token_ds.append((token_str, "".join(context_tokens)))
    return token_ds


def get_top_activating_indices_magnitude(G: Tensor, feature_idx: int, num_samples: int = 20):
    s = G[:, feature_idx]
    k = min(num_samples, s.shape[0])
    if k == 0: return [], [], [], []
    abs_vals, local_idx = torch.topk(s.abs(), k=k, largest=True)
    signed_vals = s[local_idx]
    return local_idx.tolist(), signed_vals.tolist(), abs_vals.tolist(), torch.sign(signed_vals).tolist()


def collect_feature_rows_for_layer(G: Tensor, rank: int, token_ds: Sequence[Tuple[str, str]], labels: Sequence[str],
                                   layer: int, model_name: str, concept_name: str, num_samples: int = 30,
                                   threshold: float = 0.3) -> List[Dict[str, Any]]:
    rows = []
    rank_actual = min(rank, G.shape[1])
    for k in range(rank_actual):
        idxs, _, _, _ = get_top_activating_indices_magnitude(G, k, num_samples=num_samples)
        raw_tokens = [token_ds[i][0] for i in idxs]
        labels_list = [labels[i] for i in idxs]
        num_concept_related = sum(1 for lbl in labels_list if lbl == concept_name)
        frac = num_concept_related / max(len(labels_list), 1)
        rows.append({
            "model": model_name, "concept": concept_name, "rank": rank_actual,
            "layer": layer, "feature": k, "activating_tokens": _safe_tokens(raw_tokens),
            "labels": labels_list, "num_concept_related": int(num_concept_related),
            "is_concept_realted": bool(frac >= threshold),
            "projection_top_tokens": [], "projection_bottom_tokens": [], "projection_abs_top_tokens": []
        })
    return rows


def fit_with_ridge(nmf, A: torch.Tensor, max_iter: int, patience: int = 500, base_reg: float = 1e-4,
                   tries: int = 4) -> float:
    reg = float(base_reg)
    last_err = None
    for t in range(tries):
        try:
            nmf.fit(A, max_iter, patience=patience, reg=reg)
            return reg
        except Exception as e:
            last_err = e
            reg *= 10.0
    raise RuntimeError(f"NMF failed after {tries} tries. Last error: {last_err}") from last_err

# ---------------------------------------------------------------------------
# MLP Specific Stats
# ---------------------------------------------------------------------------

def compute_mlp_layer_stats(G: torch.Tensor, is_concept: np.ndarray, is_neutral: np.ndarray, layer: int, rank: int,
                            n_samples: int, model_name: str, concept_name: str, eps: float = 1e-8) -> pd.DataFrame:
    G_np, G_abs = G.numpy(), G.abs().numpy()
    K = G.shape[1]

    c_abs_mean = G_abs[is_concept].mean(axis=0) if is_concept.any() else np.zeros(K)
    n_abs_mean = G_abs[is_neutral].mean(axis=0) if is_neutral.any() else np.zeros(K)
    c_sign_mean = G_np[is_concept].mean(axis=0) if is_concept.any() else np.zeros(K)
    n_sign_mean = G_np[is_neutral].mean(axis=0) if is_neutral.any() else np.zeros(K)

    return pd.DataFrame({
        "model": [model_name] * K, "concept": [concept_name] * K, "rank": [rank] * K,
        "n_samples": [n_samples] * K, "layer": [layer] * K, "feature": list(range(K)),
        "mean_abs_concept": c_abs_mean, "mean_abs_neutral": n_abs_mean,
        "diff_abs": c_abs_mean - n_abs_mean, "ratio_abs": c_abs_mean / (n_abs_mean + eps),
        "mean_signed_concept": c_sign_mean, "mean_signed_neutral": n_sign_mean, "diff_signed": c_sign_mean - n_sign_mean
    })


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_counts_across_concepts(counts_by_threshold: Dict[str, List[float]], concept_labels: List[str], out_path: str,
                                title: str) -> None:
    from pathlib import Path
    x = np.arange(len(concept_labels))
    plt.figure()
    labels_in_order = list(counts_by_threshold.keys())
    widths = np.linspace(0.85, 0.25, max(1, len(labels_in_order)))

    for width, lab in zip(widths, labels_in_order):
        plt.bar(x, np.asarray(counts_by_threshold[lab], dtype=float), width=width, alpha=0.6, label=lab)

    plt.xticks(x, concept_labels, rotation=25, ha="right")
    plt.ylabel("Number of features")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()