"""
Curvature of token trajectories through Llama 3.2 1B layers.

For each token t at layer l, we have hidden state h_t^(l).
After layer norm, vectors lie approximately on S^{d-1}.

Discrete curvature at layer l:
  kappa_t^(l) = ||h^(l+1) - 2*h^(l) + h^(l-1)|| / ||h^(l+1) - h^(l-1)||^2
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import load_truthfulqa, MANUAL_SENTENCES
from tqdm import tqdm


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

    # hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, d)
    hidden = torch.stack(outputs.hidden_states, dim=0).squeeze(1)  # (L+1, seq_len, d)

    # Normalize each vector to unit sphere (approximating layer norm projection)
    hidden = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)

    return hidden, inputs["input_ids"].squeeze(0)


def compute_curvature(hidden_states: torch.Tensor) -> np.ndarray:
    """
    Compute discrete curvature for each token at each interior layer.

    hidden_states: (num_layers+1, seq_len, d) — normalized
    Returns: (num_interior_layers, seq_len) curvature values
      interior layers = layers 1..L-1 (indices where we have both neighbors)
    """
    L_plus_1, seq_len, d = hidden_states.shape

    # h[l-1], h[l], h[l+1] for l in 1..L-1
    h_prev = hidden_states[:-2]   # (L-1, seq_len, d)
    h_curr = hidden_states[1:-1]  # (L-1, seq_len, d)
    h_next = hidden_states[2:]    # (L-1, seq_len, d)

    # Second difference (numerator)
    second_diff = h_next - 2 * h_curr + h_prev  # (L-1, seq_len, d)
    numerator = second_diff.norm(dim=-1)          # (L-1, seq_len)

    # Chord (denominator)
    chord = h_next - h_prev                       # (L-1, seq_len, d)
    chord_norm = chord.norm(dim=-1)               # (L-1, seq_len)
    denominator = chord_norm ** 2                  # (L-1, seq_len)

    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-10)

    kappa = numerator / denominator  # (L-1, seq_len)
    return kappa.cpu().numpy()


def _save_checkpoint(results, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(results, f)


def _load_checkpoint(path):
    import pickle
    import os
    if os.path.exists(path):
        with open(path, "rb") as f:
            results = pickle.load(f)
        print(f"Resumed from checkpoint: {len(results)} sentences already processed")
        return results, len(results)
    return [], 0


def run_experiment(layer_range: tuple = (11, 15), dataset: str = "truthfulqa", checkpoint_path: str = "checkpoint.pkl"):
    """Run the full curvature experiment.

    Args:
        layer_range: (start, end) inclusive indices (0-based) of interior layers
                     to use for focused analysis. Default is (11, 15).
        dataset: One of "truthfulqa", "manual", or "all".
                 "truthfulqa" loads from TruthfulQA (default).
                 "manual"     uses the hand-crafted MANUAL_SENTENCES list.
                 "all"        combines both.
        checkpoint_path: Path to checkpoint file for resume support.
    """
    if dataset == "truthfulqa":
        sentences = load_truthfulqa()
    elif dataset == "manual":
        sentences = MANUAL_SENTENCES
    elif dataset == "all":
        sentences = load_truthfulqa() + list(MANUAL_SENTENCES)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from 'truthfulqa', 'manual', 'all'.")

    results, start_idx = _load_checkpoint(checkpoint_path)
    sentences_to_process = sentences[start_idx:]

    if not sentences_to_process:
        print("All sentences already processed (checkpoint complete).")
        return results

    model, tokenizer, device = load_model()

    for idx, entry in enumerate(tqdm(sentences_to_process, desc="Computing curvature")):
        global_idx = start_idx + idx

        # Support both 2-tuples (manual) and 3-tuples (TruthfulQA with question)
        if len(entry) == 3:
            text, is_hallucination, question = entry
        else:
            text, is_hallucination = entry
            question = ""

        hidden, token_ids = extract_hidden_states(model, tokenizer, text, device)
        kappa = compute_curvature(hidden)  # (num_interior_layers, seq_len)

        # Determine focused layer range, clamped to available layers
        num_interior_layers = kappa.shape[0]
        layer_start = min(layer_range[0], num_interior_layers - 1)
        layer_end = min(layer_range[1], num_interior_layers - 1)
        if global_idx == 0:
            print(f"Focused analysis using interior layers {layer_start}..{layer_end} "
                  f"(of {num_interior_layers} available interior layers)")

        kappa_focused = kappa[layer_start:layer_end + 1, :]  # (focused_layers, seq_len)

        tokens = tokenizer.convert_ids_to_tokens(token_ids.cpu())

        # Build mask to filter out special tokens
        special_ids = {tid for tid in (tokenizer.bos_token_id, tokenizer.eos_token_id) if tid is not None}
        token_mask = [tid.item() not in special_ids for tid in token_ids]

        # Apply mask to filter columns (tokens) from curvature matrices
        mask_indices = [i for i, keep in enumerate(token_mask) if keep]
        kappa_filtered = kappa[:, mask_indices]
        kappa_focused_filtered = kappa_focused[:, mask_indices]
        tokens_filtered = [tokens[i] for i in mask_indices]

        # Find answer boundary: tokenize question prefix to determine where answer starts
        if question:
            q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
            # answer_start = index in filtered tokens where answer begins
            # Account for special token removal: question tokens start at index 0 of filtered
            answer_start = len(q_ids)
            # Clamp to valid range
            answer_start = min(answer_start, len(tokens_filtered))
        else:
            answer_start = 0  # No question prefix — all tokens are "answer"

        # Compute metrics on ANSWER tokens only
        if answer_start < kappa_focused_filtered.shape[1]:
            kappa_answer = kappa_focused_filtered[:, answer_start:]
        else:
            kappa_answer = kappa_focused_filtered  # Fallback: use all tokens

        results.append({
            "text": text,
            "is_hallucination": is_hallucination,
            "tokens": tokens_filtered,
            "answer_start": answer_start,
            "curvature": kappa_filtered,                                      # (L-1, seq_len_filtered)
            "curvature_focused": kappa_focused_filtered,                      # (focused_layers, seq_len_filtered)
            "mean_curvature_per_token": kappa_focused_filtered.mean(axis=0),  # (seq_len_filtered,)
            "max_curvature_per_token": kappa_focused_filtered.max(axis=0),    # (seq_len_filtered,)
            "mean_curvature_sentence": float(kappa_answer.mean()),            # Answer tokens only
            "max_curvature_sentence": float(kappa_answer.max()),              # Answer tokens only
        })

        # Checkpoint every 50 sentences
        if (global_idx + 1) % 50 == 0:
            _save_checkpoint(results, checkpoint_path)
            print(f"Checkpoint saved at {global_idx + 1} sentences.")

    return results


if __name__ == "__main__":
    import sys
    import os
    import pickle

    # Simple dataset selection via sys.argv
    dataset_choice = "truthfulqa"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("truthfulqa", "manual", "all"):
            dataset_choice = arg
        else:
            print(f"Unknown dataset argument: {sys.argv[1]!r}")
            print("Usage: python curvature.py [truthfulqa|manual|all]")
            sys.exit(1)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")
    print(f"Running experiment with dataset={dataset_choice!r}")
    results = run_experiment(dataset=dataset_choice, checkpoint_path=checkpoint_path)

    # Quick summary
    fact_means = [r["mean_curvature_sentence"] for r in results if not r["is_hallucination"]]
    hall_means = [r["mean_curvature_sentence"] for r in results if r["is_hallucination"]]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (answer tokens only)")
    print("=" * 60)
    print(f"Factual sentences  — mean curvature: {np.mean(fact_means):.6f} +/- {np.std(fact_means):.6f}")
    print(f"Hallucination sent — mean curvature: {np.mean(hall_means):.6f} +/- {np.std(hall_means):.6f}")
    print(f"Max curvature (fact): {max(r['max_curvature_sentence'] for r in results if not r['is_hallucination']):.6f}")
    print(f"Max curvature (hall): {max(r['max_curvature_sentence'] for r in results if r['is_hallucination']):.6f}")

    # Save for visualization
    results_path = os.path.join(output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed.")
