"""
Core utilities for curveline experiments.
Shared model loading, hidden state extraction, and curvature computation.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str = "unsloth/Llama-3.2-1B"):
    """Load model and tokenizer. Uses MPS on Apple Silicon."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def extract_hidden_states(model, tokenizer, text: str, device: str):
    """
    Run forward pass, return hidden states after layer norm.
    Returns: tensor of shape (num_layers+1, seq_len, hidden_dim)
      - Layer 0 = embedding, layers 1..L = transformer block outputs.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    hidden = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)

    return hidden, inputs["input_ids"].squeeze(0)


def compute_curvature(hidden_states: torch.Tensor) -> np.ndarray:
    """
    Compute discrete curvature for each token at each interior layer.

    hidden_states: (num_layers+1, seq_len, d) — normalized
    Returns: (num_interior_layers, seq_len) curvature values
    """
    h_prev = hidden_states[:-2]
    h_curr = hidden_states[1:-1]
    h_next = hidden_states[2:]

    second_diff = h_next - 2 * h_curr + h_prev
    numerator = second_diff.norm(dim=-1)

    chord = h_next - h_prev
    chord_norm = chord.norm(dim=-1)
    denominator = torch.clamp(chord_norm ** 2, min=1e-10)

    kappa = numerator / denominator
    return kappa.cpu().numpy()
