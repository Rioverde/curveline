"""
Visualization of curvature experiment results.
Generates plots for analysis of curvature vs hallucination correlation.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, accuracy_score

_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_results(path: str = None):
    if path is None:
        path = os.path.join(_DEFAULT_OUTPUT_DIR, "results.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_distribution(results, save_dir: str = None):
    """Box plot + violin for small N; histogram overlay for large N (>100)."""
    if save_dir is None:
        save_dir = _DEFAULT_OUTPUT_DIR

    fact_means = [r["mean_curvature_sentence"] for r in results if not r["is_hallucination"]]
    hall_means = [r["mean_curvature_sentence"] for r in results if r["is_hallucination"]]

    n_total = len(fact_means) + len(hall_means)

    t_stat, p_value = stats.ttest_ind(fact_means, hall_means)
    u_stat, u_pvalue = stats.mannwhitneyu(fact_means, hall_means, alternative="two-sided")

    if n_total > 100:
        # Histogram overlay for large datasets
        fig, ax = plt.subplots(figsize=(10, 6))

        bins = 40
        ax.hist(fact_means, bins=bins, alpha=0.5, color="#4CAF50", label="Factual", density=True)
        ax.hist(hall_means, bins=bins, alpha=0.5, color="#F44336", label="Hallucination", density=True)

        # Mean lines
        ax.axvline(np.mean(fact_means), color="#2E7D32", linestyle="--", linewidth=2,
                   label=f"Factual mean = {np.mean(fact_means):.4f}")
        ax.axvline(np.mean(hall_means), color="#B71C1C", linestyle="--", linewidth=2,
                   label=f"Hallucination mean = {np.mean(hall_means):.4f}")

        ax.set_xlabel("Mean Curvature")
        ax.set_ylabel("Density")
        ax.set_title(f"Mean Curvature Distribution (Layers 11-15) — N={n_total}")
        ax.legend()

        # Statistical annotations
        ax.text(0.05, 0.95, f"t-test p={p_value:.4f}\nMann-Whitney p={u_pvalue:.4f}",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat"))

        plt.tight_layout()
        plt.savefig(f"{save_dir}/curvature_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        # Box + violin for small datasets
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        data = [fact_means, hall_means]
        bp = axes[0].boxplot(data, labels=["Factual", "Hallucination"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")
        axes[0].set_ylabel("Mean Curvature")
        axes[0].set_title("Mean Curvature Distribution (Layers 11-15)")

        axes[0].text(0.05, 0.95, f"t-test p={p_value:.4f}\nMann-Whitney p={u_pvalue:.4f}",
                     transform=axes[0].transAxes, verticalalignment="top",
                     fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat"))

        parts = axes[1].violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#4CAF50", "#F44336"][i])
            pc.set_alpha(0.7)
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Factual", "Hallucination"])
        axes[1].set_ylabel("Mean Curvature")
        axes[1].set_title("Curvature Violin Plot (Layers 11-15)")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/curvature_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("Saved: curvature_distribution.png")
    return t_stat, p_value, u_stat, u_pvalue


def plot_layer_profile(results, save_dir: str = None):
    """Mean curvature per layer for factual vs hallucination sentences (all layers)."""
    if save_dir is None:
        save_dir = _DEFAULT_OUTPUT_DIR

    fact_layers = []
    hall_layers = []

    for r in results:
        # curvature shape: (num_interior_layers, seq_len) — mean over tokens
        layer_means = r["curvature"].mean(axis=1)  # (num_interior_layers,)
        if r["is_hallucination"]:
            hall_layers.append(layer_means)
        else:
            fact_layers.append(layer_means)

    fact_layers = np.array(fact_layers)
    hall_layers = np.array(hall_layers)

    # Interior layers start at layer 2 (index 0 = layer 2, etc.)
    num_layers = fact_layers.shape[1]
    layers = np.arange(2, num_layers + 2)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Highlight the focused layer range (11-15)
    ax.axvspan(11, 15, alpha=0.12, color="blue", label="Analysis range (L11-L15)")

    # Mean +/- std
    ax.plot(layers, fact_layers.mean(axis=0), "g-o", label="Factual", linewidth=2)
    ax.fill_between(layers,
                    fact_layers.mean(axis=0) - fact_layers.std(axis=0),
                    fact_layers.mean(axis=0) + fact_layers.std(axis=0),
                    alpha=0.2, color="green")

    ax.plot(layers, hall_layers.mean(axis=0), "r-s", label="Hallucination", linewidth=2)
    ax.fill_between(layers,
                    hall_layers.mean(axis=0) - hall_layers.std(axis=0),
                    hall_layers.mean(axis=0) + hall_layers.std(axis=0),
                    alpha=0.2, color="red")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Curvature")
    ax.set_title("Curvature Profile Across All Layers (shaded: analysis region L11-L15)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/curvature_layer_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: curvature_layer_profile.png")


def plot_heatmap(results, save_dir: str = None):
    """Heatmap of curvature per token per layer for a few example sentences."""
    if save_dir is None:
        save_dir = _DEFAULT_OUTPUT_DIR

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    facts = [r for r in results if not r["is_hallucination"]]
    halls = [r for r in results if r["is_hallucination"]]

    n_total = len(facts) + len(halls)

    if n_total > 100:
        # Random sample with fixed seed for reproducibility (avoids cherry-picking)
        rng = np.random.default_rng(42)
        fact_idxs = rng.choice(len(facts), size=min(2, len(facts)), replace=False)
        hall_idxs = rng.choice(len(halls), size=min(2, len(halls)), replace=False)
        selected_facts = [(facts[i], f"Factual (sample {j+1})") for j, i in enumerate(fact_idxs)]
        selected_halls = [(halls[i], f"Hallucination (sample {j+1})") for j, i in enumerate(hall_idxs)]
        examples = selected_facts + selected_halls
    else:
        # Sort by max curvature for small datasets
        facts.sort(key=lambda r: r["max_curvature_sentence"], reverse=True)
        halls.sort(key=lambda r: r["max_curvature_sentence"], reverse=True)
        examples = [
            (facts[0], "Factual (highest kappa)"),
            (facts[-1], "Factual (lowest kappa)"),
            (halls[0], "Hallucination (highest kappa)"),
            (halls[-1], "Hallucination (lowest kappa)"),
        ]

    for ax, (r, title) in zip(axes.flat, examples):
        kappa = r["curvature_focused"]  # (num_focused_layers, seq_len) — layers 11-15
        tokens = r["tokens"]  # already filtered (no special tokens)

        # Truncate long sequences for readability
        max_tokens = 25
        if len(tokens) > max_tokens:
            kappa = kappa[:, :max_tokens]
            tokens = tokens[:max_tokens]

        # Y-axis labels show actual layer numbers (11, 12, ..., 11+n-1)
        num_focused = kappa.shape[0]
        layer_labels = [f"L{11 + i}" for i in range(num_focused)]

        sns.heatmap(kappa, ax=ax, cmap="YlOrRd", xticklabels=tokens,
                    yticklabels=layer_labels)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Token")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/curvature_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: curvature_heatmaps.png")


def plot_scatter(results, save_dir: str = None):
    """Scatter plot: mean vs max curvature, colored by label."""
    if save_dir is None:
        save_dir = _DEFAULT_OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        color = "#F44336" if r["is_hallucination"] else "#4CAF50"
        marker = "x" if r["is_hallucination"] else "o"
        ax.scatter(r["mean_curvature_sentence"], r["max_curvature_sentence"],
                   c=color, marker=marker, s=60, alpha=0.8)

    ax.set_xlabel("Mean Curvature")
    ax.set_ylabel("Max Curvature")
    ax.set_title("Mean vs Max Curvature per Sentence (Layers 11-15)")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50", markersize=10, label="Factual"),
        Line2D([0], [0], marker="x", color="#F44336", markersize=10, label="Hallucination"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/curvature_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: curvature_scatter.png")


def plot_roc(results, save_dir: str = None):
    """ROC curve using mean_curvature_sentence as classifier score."""
    if save_dir is None:
        save_dir = _DEFAULT_OUTPUT_DIR

    scores = np.array([r["mean_curvature_sentence"] for r in results])
    labels = np.array([int(r["is_hallucination"]) for r in results])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Optimal threshold via Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Accuracy at optimal threshold
    preds = (scores >= optimal_threshold).astype(int)
    acc = accuracy_score(labels, preds)

    print(f"\nROC Analysis:")
    print(f"  AUC:                  {roc_auc:.4f}")
    print(f"  Optimal threshold:    {optimal_threshold:.6f} (Youden's J)")
    print(f"  Accuracy at optimal:  {acc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red", s=100, zorder=5,
               label=f"Optimal threshold = {optimal_threshold:.4f}\n(accuracy = {acc:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Mean Curvature as Hallucination Classifier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/curvature_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: curvature_roc.png")

    return roc_auc, optimal_threshold, acc


def print_stats(results):
    """Print detailed statistical analysis."""
    fact_means = [r["mean_curvature_sentence"] for r in results if not r["is_hallucination"]]
    hall_means = [r["mean_curvature_sentence"] for r in results if r["is_hallucination"]]
    fact_maxs = [r["max_curvature_sentence"] for r in results if not r["is_hallucination"]]
    hall_maxs = [r["max_curvature_sentence"] for r in results if r["is_hallucination"]]

    n_fact = len(fact_means)
    n_hall = len(hall_means)
    n_total = n_fact + n_hall

    # ROC metrics
    scores = np.array([r["mean_curvature_sentence"] for r in results])
    labels = np.array([int(r["is_hallucination"]) for r in results])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    preds = (scores >= optimal_threshold).astype(int)
    acc = accuracy_score(labels, preds)

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"Analysis focused on layers 11-15 | Special tokens filtered")
    print(f"Dataset: {n_fact} factual, {n_hall} hallucination sentences (N={n_total} total)")

    # Descriptive stats
    print(f"\nMean curvature:")
    print(f"  Factual:       {np.mean(fact_means):.6f} +/- {np.std(fact_means):.6f}")
    print(f"  Hallucination: {np.mean(hall_means):.6f} +/- {np.std(hall_means):.6f}")

    print(f"\nMax curvature:")
    print(f"  Factual:       {np.mean(fact_maxs):.6f} +/- {np.std(fact_maxs):.6f}")
    print(f"  Hallucination: {np.mean(hall_maxs):.6f} +/- {np.std(hall_maxs):.6f}")

    # Tests
    t_stat, p_value = stats.ttest_ind(fact_means, hall_means)
    u_stat, u_pvalue = stats.mannwhitneyu(fact_means, hall_means, alternative="two-sided")
    d = (np.mean(hall_means) - np.mean(fact_means)) / np.sqrt(
        (np.std(fact_means) ** 2 + np.std(hall_means) ** 2) / 2
    )

    print(f"\nStatistical tests (mean curvature):")
    print(f"  t-test:        t={t_stat:.4f}, p={p_value:.4f}")
    print(f"  Mann-Whitney:  U={u_stat:.1f}, p={u_pvalue:.4f}")
    print(f"  Cohen's d:     {d:.4f}")

    # ROC / classification metrics
    print(f"\nClassification metrics (mean curvature as score):")
    print(f"  ROC AUC:              {roc_auc:.4f}")
    print(f"  Optimal threshold:    {optimal_threshold:.6f} (Youden's J)")
    print(f"  Accuracy at optimal:  {acc:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        direction = "HIGHER" if np.mean(hall_means) > np.mean(fact_means) else "LOWER"
        print(f"  Significant difference (p<0.05): hallucinations have {direction} curvature")
    else:
        print(f"  No significant difference at p<0.05")

    if n_total > 100:
        print(f"  NOTE: With N={n_total}, p-values become trivially small for any real effect.")
        print(f"  Emphasize Cohen's d as the primary effect size measure.")

    if abs(d) > 0.8:
        print(f"  Large effect size (|d|={abs(d):.4f} > 0.8)")
    elif abs(d) > 0.5:
        print(f"  Medium effect size (|d|={abs(d):.4f} > 0.5)")
    elif abs(d) > 0.2:
        print(f"  Small effect size (|d|={abs(d):.4f} > 0.2)")
    else:
        print(f"  Negligible effect size (|d|={abs(d):.4f})")


if __name__ == "__main__":
    output_dir = _DEFAULT_OUTPUT_DIR
    results = load_results(os.path.join(output_dir, "results.pkl"))
    t, p, u, up = plot_distribution(results, save_dir=output_dir)
    plot_layer_profile(results, save_dir=output_dir)
    plot_heatmap(results, save_dir=output_dir)
    plot_scatter(results, save_dir=output_dir)
    plot_roc(results, save_dir=output_dir)
    print_stats(results)
