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

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded: {config.num_hidden_layers} layers, "
          f"d_model={config.hidden_size}, d_mlp={config.intermediate_size}")

    return LocalModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        n_layers=config.num_hidden_layers,
        d_model=config.hidden_size,
        d_mlp=config.intermediate_size
    )