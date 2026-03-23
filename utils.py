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

def get_embedding_matrix(model) -> torch.Tensor:
    if hasattr(model, "W_E") and torch.is_tensor(model.W_E): return model.W_E
    if hasattr(model, "embed") and hasattr(model.embed, "W_E") and torch.is_tensor(
        model.embed.W_E): return model.embed.W_E
    raise RuntimeError("Could not locate token embedding matrix.")


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
# Embedding Specific Stats
# ---------------------------------------------------------------------------

def build_token_label_codes(concept_name: str, dataset, tokenizer) -> Dict[int, int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    origin = {}
    for sent, lbl in dataset:
        if not sent: continue
        for tid in tokenizer(sent, add_special_tokens=False)["input_ids"]:
            tid = int(tid)
            if tid in special: continue
            origin.setdefault(tid, set()).add(
                "neutral" if lbl == "Neutral" else "concept" if lbl == concept_name else str(lbl))

    token_to_code = {}
    for tid, s in origin.items():
        if "concept" in s and "neutral" in s:
            token_to_code[tid] = 2
        elif "concept" in s:
            token_to_code[tid] = 1
        else:
            token_to_code[tid] = 0
    return token_to_code


def compute_embedding_stats(F_tok: torch.Tensor, vprime_token_ids: List[int], token_to_code: Dict[int, int],
                            concept_name: str, nonzero_eps: float = 1e-12, ratio_eps: float = 1e-12) -> pd.DataFrame:
    Vp, K = F_tok.shape
    codes = np.zeros(Vp, dtype=np.int8)
    for i, tid in enumerate(vprime_token_ids): codes[i] = token_to_code.get(tid, 0)
    codes_t = torch.from_numpy(codes)

    rows = []
    for k in range(K):
        w = F_tok[:, k].abs()
        nz_mask = w > nonzero_eps
        w_nz, c_nz = (w[nz_mask], codes_t[nz_mask]) if nz_mask.any() else (w.new_zeros((0,)), codes_t.new_zeros((0,)))

        m_neu, m_con, m_both = (c_nz == 0), (c_nz == 1), (c_nz == 2)
        mass_neu = float(w_nz[m_neu].sum()) if m_neu.any() else 0.0
        mass_con = float(w_nz[m_con].sum()) if m_con.any() else 0.0

        rows.append({
            "concept": concept_name, "feature": k,
            "num_concept": int(m_con.sum()), "num_neutral": int(m_neu.sum()), "num_both": int(m_both.sum()),
            "mass_concept": mass_con, "mass_neutral": mass_neu,
            "mass_both": float(w_nz[m_both].sum()) if m_both.any() else 0.0,
            "mass_ratio_cn": (mass_con + ratio_eps) / (mass_neu + ratio_eps),
        })
    return pd.DataFrame(rows)


def collect_feature_rows_for_embeddings(F_tok: torch.Tensor, vprime_token_ids: Sequence[int],
                                        token_label_map: Dict[int, str], model, model_name: str, concept_name: str,
                                        rank: int, max_activating: int = 200) -> List[Dict[str, Any]]:
    n_vocabprime, K = F_tok.shape
    token_strs = model.to_str_tokens(torch.tensor(list(vprime_token_ids), dtype=torch.long))
    vprime_tok_strs = _safe_tokens([str(t[0][0]) if isinstance(t, list) and len(t) > 0 and isinstance(t[0],
                                                                                                      list) else str(
        t[0]) if isinstance(t, list) else str(t) for t in token_strs])

    rows = []
    for k in range(min(rank, K)):
        col = F_tok[:, k]
        nz = torch.where(col.abs() > 1e-12)[0].tolist()
        nz_sorted = sorted(nz, key=lambda i: float(col.abs()[i]), reverse=True)[:max_activating] if len(nz) > 1 else nz

        act_tokens = [vprime_tok_strs[i] for i in nz_sorted]
        labels_list = [token_label_map.get(int(vprime_token_ids[i]), "Neutral") for i in nz_sorted]
        frac = sum(1 for lab in labels_list if lab in (concept_name, "both")) / max(len(labels_list), 1)

        rows.append({
            "model": model_name, "concept": concept_name, "rank": min(rank, K), "feature": k,
            "num_activating_tokens_all": len(nz), "activating_tokens": act_tokens, "labels": labels_list,
            "num_concept_related": sum(1 for lab in labels_list if lab in (concept_name, "both")),
            "is_concept_realted": bool(frac >= 0.7),
            "projection_top_tokens": [], "projection_bottom_tokens": [], "projection_abs_top_tokens": []
        })
    return rows


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