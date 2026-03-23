import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel
from dataclasses import dataclass

@dataclass
class LocalModel:
    model: PreTrainedModel
    tokenizer: AutoTokenizer
    config: AutoConfig
    device: str
    n_layers: int
    d_model: int
    d_mlp: int


def load_local_model(model_path: str, device: str = "cpu") -> LocalModel:
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        output_hidden_states=True,
        attn_implementation='eager',
        torch_dtype=torch.float32,
    )
    model.eval().to(device)

    try:
        print("Attempting to load tokenizer from official repo...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)
    except Exception as e:
        print(f"Could not load from repo, trying local with fallback: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if hasattr(tokenizer, 'extra_special_tokens') and isinstance(tokenizer.extra_special_tokens, list):
        tokenizer.extra_special_tokens = {t: t for t in tokenizer.extra_special_tokens}

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded: {config.num_hidden_layers} layers, d_model={config.hidden_size}")
    return LocalModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        n_layers=config.num_hidden_layers,
        d_model=config.hidden_size,
        d_mlp=config.intermediate_size
    )