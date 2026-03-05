"""Visualization for the early exit experiment."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_results(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "results.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_convergence_by_complexity(results, save_dir="."):
    """Box plot: convergence layer distribution for simple/medium/complex."""
    data = {}
    for complexity in ["simple", "medium", "complex"]:
        group = [r for r in results if r["complexity"] == complexity]
        # Collect ALL per-token convergence layers
        all_conv = []
        for r in group:
            all_conv.extend(r["convergence_layers"])
        data[complexity] = all_conv

    fig, ax = plt.subplots(figsize=(8, 6))

    positions = [1, 2, 3]
    labels = ["Simple", "Medium", "Complex"]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    bp = ax.boxplot([data["simple"], data["medium"], data["complex"]],
                     positions=positions, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Convergence Layer")
    ax.set_title("Token Convergence Layer by Text Complexity")
    ax.grid(True, alpha=0.3, axis="y")

    # ANOVA test
    f_stat, p_value = stats.f_oneway(data["simple"], data["medium"], data["complex"])
    ax.text(0.05, 0.95, f"ANOVA F={f_stat:.2f}, p={p_value:.4f}",
            transform=ax.transAxes, verticalalignment="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/convergence_by_complexity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: convergence_by_complexity.png")

    return f_stat, p_value


def plot_curvature_decay(results, save_dir="."):
    """Line plot: mean curvature per layer, grouped by complexity."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"simple": "#4CAF50", "medium": "#FF9800", "complex": "#F44336"}

    for complexity in ["simple", "medium", "complex"]:
        group = [r for r in results if r["complexity"] == complexity]
        layer_curves = np.array([r["curvature_by_layer"] for r in group])

        num_layers = layer_curves.shape[1]
        layers = np.arange(1, num_layers + 1)

        mean_curve = layer_curves.mean(axis=0)
        std_curve = layer_curves.std(axis=0)

        ax.plot(layers, mean_curve, "-o", color=colors[complexity],
                label=complexity.capitalize(), linewidth=2, markersize=4)
        ax.fill_between(layers, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=colors[complexity])

    ax.set_xlabel("Interior Layer")
    ax.set_ylabel("Mean Curvature")
    ax.set_title("Curvature Decay Across Layers by Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/curvature_decay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: curvature_decay.png")


def plot_token_convergence_heatmap(results, save_dir="."):
    """Heatmap with convergence markers for one example per complexity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, complexity in zip(axes, ["simple", "medium", "complex"]):
        group = [r for r in results if r["complexity"] == complexity]
        # Pick the first example
        r = group[0]
        kappa = r["curvature"]
        tokens = r["tokens"]
        conv_layers = r["convergence_layers"]

        # Truncate long sequences
        max_tokens = 20
        if len(tokens) > max_tokens:
            kappa = kappa[:, :max_tokens]
            tokens = tokens[:max_tokens]
            conv_layers = conv_layers[:max_tokens]

        num_layers = kappa.shape[0]
        layer_labels = [f"L{i+1}" for i in range(num_layers)]

        sns.heatmap(kappa, ax=ax, cmap="YlOrRd", xticklabels=tokens,
                    yticklabels=layer_labels)

        # Mark convergence layer for each token
        for t, cl in enumerate(conv_layers):
            if cl < num_layers:
                ax.plot(t + 0.5, cl + 0.5, "bD", markersize=5)

        ax.set_title(f"{complexity.capitalize()}: {r['text'][:40]}...", fontsize=9)
        ax.set_xlabel("Token")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/token_convergence_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: token_convergence_heatmap.png")


def plot_convergence_histogram(results, save_dir="."):
    """Histogram of convergence layers across all tokens, grouped by complexity."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"simple": "#4CAF50", "medium": "#FF9800", "complex": "#F44336"}

    for complexity in ["simple", "medium", "complex"]:
        group = [r for r in results if r["complexity"] == complexity]
        all_conv = []
        for r in group:
            all_conv.extend(r["convergence_layers"])

        ax.hist(all_conv, bins=range(0, max(all_conv) + 2), alpha=0.5,
                color=colors[complexity], label=complexity.capitalize(), density=True)

    ax.set_xlabel("Convergence Layer")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Token Convergence Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/convergence_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: convergence_histogram.png")


def print_stats(results):
    """Print statistical analysis of convergence layers."""
    print("\n" + "=" * 60)
    print("EARLY EXIT — STATISTICAL ANALYSIS")
    print("=" * 60)

    data = {}
    for complexity in ["simple", "medium", "complex"]:
        group = [r for r in results if r["complexity"] == complexity]
        all_conv = []
        for r in group:
            all_conv.extend(r["convergence_layers"])
        data[complexity] = all_conv

        mean_sentence = [r["mean_convergence_layer"] for r in group]
        print(f"\n{complexity.capitalize()}:")
        print(f"  Per-token:    mean={np.mean(all_conv):.2f} +/- {np.std(all_conv):.2f} (n={len(all_conv)} tokens)")
        print(f"  Per-sentence: mean={np.mean(mean_sentence):.2f} +/- {np.std(mean_sentence):.2f} (n={len(mean_sentence)} texts)")

    # ANOVA
    f_stat, p_value = stats.f_oneway(data["simple"], data["medium"], data["complex"])
    print(f"\nANOVA (per-token convergence):")
    print(f"  F={f_stat:.4f}, p={p_value:.6f}")

    # Pairwise t-tests
    for a, b in [("simple", "medium"), ("medium", "complex"), ("simple", "complex")]:
        t, p = stats.ttest_ind(data[a], data[b])
        print(f"  {a} vs {b}: t={t:.4f}, p={p:.6f}")

    # Kruskal-Wallis (non-parametric)
    h_stat, kw_p = stats.kruskal(data["simple"], data["medium"], data["complex"])
    print(f"\nKruskal-Wallis: H={h_stat:.4f}, p={kw_p:.6f}")

    # Effect sizes
    for a, b in [("simple", "complex"),]:
        d = (np.mean(data[b]) - np.mean(data[a])) / np.sqrt(
            (np.std(data[a])**2 + np.std(data[b])**2) / 2)
        print(f"\nCohen's d (simple vs complex): {d:.4f}")
        if abs(d) > 0.8:
            print("  Large effect size")
        elif abs(d) > 0.5:
            print("  Medium effect size")
        elif abs(d) > 0.2:
            print("  Small effect size")
        else:
            print("  Negligible effect size")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    results = load_results()
    plot_convergence_by_complexity(results, save_dir=output_dir)
    plot_curvature_decay(results, save_dir=output_dir)
    plot_token_convergence_heatmap(results, save_dir=output_dir)
    plot_convergence_histogram(results, save_dir=output_dir)
    print_stats(results)
