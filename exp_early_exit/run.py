"""
Early Exit Experiment — where does token processing converge?

Hypothesis: curvature is high in early layers (active processing),
then drops and stabilizes. The convergence layer varies by text complexity.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from tqdm import tqdm
from core import load_model, extract_hidden_states, compute_curvature

TEXTS = {
    "simple": [
        "The cat sat on the mat.",
        "Water is wet.",
        "The sky is blue.",
        "Dogs are animals.",
        "One plus one equals two.",
        "The sun is bright.",
        "Fish live in water.",
        "Birds can fly.",
        "Snow is cold.",
        "Fire is hot.",
    ],
    "medium": [
        "The process of photosynthesis converts carbon dioxide and water into glucose and oxygen.",
        "The French Revolution of 1789 fundamentally altered the political landscape of Europe.",
        "Quantum mechanics describes the behavior of particles at the subatomic level.",
        "The human immune system produces antibodies to fight specific pathogens.",
        "Shakespeare's plays explore themes of power, jealousy, and the human condition.",
        "The theory of plate tectonics explains the movement of Earth's lithospheric plates.",
        "Neural networks learn to recognize patterns by adjusting connection weights during training.",
        "The Industrial Revolution transformed manufacturing from manual labor to machine-based production.",
        "Mitochondrial DNA is inherited exclusively from the maternal lineage in most organisms.",
        "The double helix structure of DNA was discovered by Watson and Crick in 1953.",
    ],
    "complex": [
        "The renormalization group in quantum field theory provides a systematic framework for understanding how physical systems behave across different energy scales.",
        "Gödel's incompleteness theorems demonstrate that any sufficiently powerful axiomatic system contains statements that are true but unprovable within that system.",
        "The holographic principle suggests that the information contained within a volume of space can be described by a theory operating on the boundary of that region.",
        "Epigenetic modifications such as DNA methylation and histone acetylation regulate gene expression without altering the underlying nucleotide sequence.",
        "The Riemann hypothesis, concerning the distribution of non-trivial zeros of the zeta function, remains one of the most important unsolved problems in mathematics.",
        "Topological quantum computing exploits anyonic braiding statistics to perform fault-tolerant quantum computation resistant to local perturbation.",
        "The adaptive immune system employs somatic hypermutation and clonal selection to generate high-affinity antibodies against novel pathogens.",
        "Category theory provides a unifying mathematical framework that abstracts the common structure underlying diverse mathematical constructions.",
        "The arrow of time emerges from the second law of thermodynamics and the low-entropy initial conditions of the observable universe.",
        "Consciousness remains poorly understood, with competing theories including integrated information theory and global workspace theory offering different computational accounts.",
    ],
}


def find_convergence_layer(curvature_per_layer, rel_change_threshold=0.05):
    """
    Find the layer where token processing stabilizes.

    Stabilization = the relative change in curvature between consecutive layers
    drops below threshold AND stays below for all remaining layers.

    |kappa[l+1] - kappa[l]| / kappa[l] < threshold

    Args:
        curvature_per_layer: 1D array of curvature values per layer for one token
        rel_change_threshold: max relative change to consider "stable"

    Returns:
        convergence layer index, or len-1 if never converges
    """
    n = len(curvature_per_layer)
    if n < 2:
        return 0

    # Compute relative change between consecutive layers
    rel_changes = np.zeros(n - 1)
    for i in range(n - 1):
        if curvature_per_layer[i] > 1e-10:
            rel_changes[i] = abs(curvature_per_layer[i + 1] - curvature_per_layer[i]) / curvature_per_layer[i]
        else:
            rel_changes[i] = 0.0

    # Find first layer where all subsequent changes are below threshold
    for layer in range(len(rel_changes)):
        if np.all(rel_changes[layer:] < rel_change_threshold):
            return layer

    return n - 1


def run_experiment():
    """Run the early exit experiment."""
    model, tokenizer, device = load_model()

    special_ids = {tid for tid in (tokenizer.bos_token_id, tokenizer.eos_token_id) if tid is not None}

    results = []
    all_texts = [(text, complexity) for complexity, texts in TEXTS.items() for text in texts]

    for text, complexity in tqdm(all_texts, desc="Processing texts"):
        hidden, token_ids = extract_hidden_states(model, tokenizer, text, device)
        kappa = compute_curvature(hidden)  # (num_interior_layers, seq_len)

        tokens = tokenizer.convert_ids_to_tokens(token_ids.cpu())

        # Filter special tokens
        token_mask = [tid.item() not in special_ids for tid in token_ids]
        mask_indices = [i for i, keep in enumerate(token_mask) if keep]
        kappa_filtered = kappa[:, mask_indices]
        tokens_filtered = [tokens[i] for i in mask_indices]

        # Find convergence layer for each token
        convergence_layers = []
        for t in range(kappa_filtered.shape[1]):
            token_curvature = kappa_filtered[:, t]  # curvature across all interior layers
            conv_layer = find_convergence_layer(token_curvature)
            convergence_layers.append(conv_layer)

        results.append({
            "text": text,
            "complexity": complexity,
            "tokens": tokens_filtered,
            "curvature": kappa_filtered,
            "convergence_layers": convergence_layers,
            "mean_convergence_layer": float(np.mean(convergence_layers)) if convergence_layers else 0.0,
            "curvature_by_layer": kappa_filtered.mean(axis=1),  # mean over tokens per layer
        })

    return results


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    results = run_experiment()

    # Summary
    print("\n" + "=" * 60)
    print("EARLY EXIT — CONVERGENCE SUMMARY")
    print("=" * 60)
    for complexity in ["simple", "medium", "complex"]:
        group = [r for r in results if r["complexity"] == complexity]
        conv_layers = [r["mean_convergence_layer"] for r in group]
        print(f"{complexity:>8}: mean convergence layer = {np.mean(conv_layers):.2f} +/- {np.std(conv_layers):.2f}")

    # Save
    results_path = os.path.join(output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")
