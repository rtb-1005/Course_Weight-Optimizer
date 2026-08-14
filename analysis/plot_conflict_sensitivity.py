"""Generate publication-style sensitivity plots for course conflicts.

Run from the repository root with a Python environment that has matplotlib:

    python analysis/plot_conflict_sensitivity.py

The script keeps the current proxy probability model fixed and changes only
the feasibility graph. This isolates the direct effect of course conflicts.
"""

from __future__ import annotations

import itertools
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Course_Weight-Optimizer"
sys.path.insert(0, str(SRC))

import main as optimizer  # noqa: E402
from utils import CourseState, GlobalState, load_desired_courses, load_global_state  # noqa: E402


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
}


def build_model(ids: Sequence[str], utilities: Dict[str, float], state: GlobalState):
    total_bidders = sum(float(course.bidders) for course in state.courses.values())
    s_bar = optimizer.clamp(
        total_bidders / float(state.grade_size),
        1.0,
        optimizer.BUDGET / optimizer.MIN_BID,
    )
    s = optimizer.clamp(
        s_bar * optimizer.S_MULTS[optimizer.DESIGN_INDEX],
        1.0,
        optimizer.BUDGET / optimizer.MIN_BID,
    )
    mu = optimizer.BUDGET / max(s, optimizer.EPS)
    predicted = optimizer.predict_final_bidders(state, s)
    alphas = {
        cid: optimizer.compute_alpha(
            state.courses[cid].capacity,
            predicted[cid] / float(state.courses[cid].capacity),
            mu,
        )
        for cid in ids
    }
    return list(ids), utilities, alphas


def score_subset(subset: Sequence[str], utilities: Dict[str, float], alphas: Dict[str, float]) -> float:
    bids = optimizer.waterfill_allocate(
        list(subset), utilities, alphas, optimizer.BUDGET, optimizer.MIN_BID
    )
    return sum(
        utilities[cid] * optimizer.proxy_prob(bids[cid], alphas[cid])
        for cid in subset
    )


def subset_scores(ids: Sequence[str], utilities: Dict[str, float], alphas: Dict[str, float]):
    max_size = min(int(optimizer.BUDGET // optimizer.MIN_BID), len(ids))
    return [
        (subset, score_subset(subset, utilities, alphas))
        for size in range(1, max_size + 1)
        for subset in itertools.combinations(ids, size)
    ]


def best_independent(candidates, conflicts: Iterable[Tuple[str, str]]):
    best = (tuple(), 0.0)
    for subset, score in candidates:
        if optimizer.is_conflict_free(subset, conflicts) and score > best[1]:
            best = (subset, score)
    return best


def sample_data():
    desired = load_desired_courses(str(SRC / "desired_courses.json"))
    state = load_global_state(str(SRC / "global_state.json"))
    utilities = {
        pref.course_id: pref.utility
        for pref in desired.preferences
        if pref.course_id in state.courses and pref.utility > 0
    }
    ids = list(utilities)
    return ids, utilities, state, build_model(ids, utilities, state)[2]


def synthetic_data(seed: int, n: int = 10):
    rng = random.Random(seed)
    courses: Dict[str, CourseState] = {}
    utilities: Dict[str, float] = {}
    for index in range(n):
        cid = f"SYN_{index + 1:02d}"
        capacity = rng.choice([20, 25, 30, 35, 40])
        bidders = rng.randint(int(capacity * 1.05), int(capacity * 2.1))
        courses[cid] = CourseState(cid, capacity, bidders)
        utilities[cid] = round(rng.uniform(1.0, 10.0), 2)
    state = GlobalState(grade_size=126, courses=courses)
    ids, utilities, alphas = build_model(list(utilities), utilities, state)
    return ids, utilities, state, alphas


def random_edges(n: int, p: float, rng: random.Random):
    return [
        pair
        for pair in itertools.combinations(range(n), 2)
        if rng.random() < p
    ]


def sample_exact_curves(ids, utilities, alphas):
    candidates = subset_scores(ids, utilities, alphas)
    baseline = best_independent(candidates, [])
    pairs = list(itertools.combinations(range(len(ids)), 2))
    invalid_curve = []
    ratio_curve = []
    for p in (0.01, 0.05, 0.10, 0.20, 0.30):
        expected_invalid = 0.0
        expected_ratio = 0.0
        for bits in range(1 << len(pairs)):
            edge_count = bits.bit_count()
            probability = p**edge_count * (1.0 - p) ** (len(pairs) - edge_count)
            edges = [
                (ids[pairs[i][0]], ids[pairs[i][1]])
                for i in range(len(pairs))
                if bits & (1 << i)
            ]
            if not optimizer.is_conflict_free(baseline[0], edges):
                expected_invalid += probability
            expected_ratio += probability * best_independent(candidates, edges)[1] / baseline[1]
        invalid_curve.append(expected_invalid)
        ratio_curve.append(expected_ratio)
    return (0.01, 0.05, 0.10, 0.20, 0.30), baseline, invalid_curve, ratio_curve


def analytic_invalid_curve():
    """Exact invalidation probability for any old portfolio size m."""
    course_counts = list(range(2, 21))
    probabilities = (0.01, 0.05, 0.10)
    curves = {
        p: [1.0 - (1.0 - p) ** (m * (m - 1) // 2) for m in course_counts]
        for p in probabilities
    }
    return course_counts, probabilities, curves


def generalized_monte_carlo():
    """Simulate several portfolio sizes, not only the repository example."""
    optimizer.BISect_ITERS = 20
    course_counts = (3, 5, 8, 10, 12)
    probabilities = (0.05, 0.10, 0.20)
    ratio_samples = {(p, n): [] for p in probabilities for n in course_counts}
    selected_samples = {(p, n): [] for p in probabilities for n in course_counts}
    invalid_samples = {(p, n): [] for p in probabilities for n in course_counts}
    topology_rng = random.Random(1005)

    for n in course_counts:
        for seed in range(8):
            ids, utilities, _, alphas = synthetic_data(seed + 100 * n, n=n)
            candidates = subset_scores(ids, utilities, alphas)
            baseline = best_independent(candidates, [])
            for p in probabilities:
                seed_invalid = []
                seed_ratios = []
                seed_selected = []
                for _ in range(40):
                    edges = [
                        (ids[left], ids[right])
                        for left, right in random_edges(len(ids), p, topology_rng)
                    ]
                    feasible = optimizer.is_conflict_free(baseline[0], edges)
                    constrained = best_independent(candidates, edges)
                    seed_invalid.append(0 if feasible else 1)
                    seed_ratios.append(constrained[1] / baseline[1])
                    seed_selected.append(len(constrained[0]))
                invalid_samples[(p, n)].append(mean(seed_invalid))
                ratio_samples[(p, n)].append(mean(seed_ratios))
                selected_samples[(p, n)].append(mean(seed_selected))
    return course_counts, probabilities, invalid_samples, ratio_samples, selected_samples


def mean_ci(values: Sequence[float]):
    if len(values) < 2:
        return mean(values), 0.0
    return mean(values), 1.96 * stdev(values) / (len(values) ** 0.5)


def topology_losses(ids, utilities, alphas):
    candidates = subset_scores(ids, utilities, alphas)
    baseline = best_independent(candidates, [])[1]
    scenarios = {
        "One pair": [(0, 1)],
        "Chain": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "Star": [(1, 0), (1, 2), (1, 3), (1, 4)],
        "Clique": list(itertools.combinations(range(5), 2)),
    }
    losses = {}
    for name, edges in scenarios.items():
        named_edges = [(ids[left], ids[right]) for left, right in edges]
        losses[name] = 1.0 - best_independent(candidates, named_edges)[1] / baseline
    return losses


def pair_loss_matrix(ids, utilities, alphas):
    candidates = subset_scores(ids, utilities, alphas)
    baseline = best_independent(candidates, [])[1]
    matrix = [[float("nan") for _ in ids] for _ in ids]
    for left, right in itertools.combinations(range(len(ids)), 2):
        edges = [(ids[left], ids[right])]
        loss = 1.0 - best_independent(candidates, edges)[1] / baseline
        matrix[left][right] = loss
        matrix[right][left] = loss
    return matrix


def draw():
    ids, utilities, _, alphas = sample_data()
    sample_probabilities, baseline, sample_invalid, sample_ratio = sample_exact_curves(ids, utilities, alphas)
    course_counts, probabilities, invalid_samples, ratio_samples, selected_samples = generalized_monte_carlo()
    analytic_counts, analytic_probabilities, analytic_curves = analytic_invalid_curve()
    topology = topology_losses(ids, utilities, alphas)
    matrix = pair_loss_matrix(ids, utilities, alphas)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    ax_invalid, ax_ratio, ax_selected, ax_topology, ax_heatmap, ax_unused = axes.ravel()
    ax_unused.axis("off")

    analytic_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["vermillion"]]
    for p, color in zip(analytic_probabilities, analytic_colors):
        ax_invalid.plot(
            analytic_counts,
            analytic_curves[p],
            linewidth=2.2,
            color=color,
            label=f"p={p:.0%}",
        )
    ax_invalid.set_title("A  Any portfolio size: feasibility failure")
    ax_invalid.set_xlabel("Courses selected by old optimizer (m)")
    ax_invalid.set_ylabel("Probability portfolio is invalid")
    ax_invalid.set_ylim(0, 1.05)
    ax_invalid.set_xticks([2, 5, 8, 10, 12, 15, 20])
    ax_invalid.legend(frameon=False, fontsize=8)

    for p, color in zip(probabilities, [PALETTE["blue"], PALETTE["green"], PALETTE["vermillion"]]):
        means = [mean(ratio_samples[(p, n)]) for n in course_counts]
        errors = [mean_ci(ratio_samples[(p, n)])[1] for n in course_counts]
        ax_ratio.errorbar(
            course_counts,
            means,
            yerr=errors,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=f"p={p:.0%}",
        )
    ax_ratio.set_title("B  Monte Carlo: retained proxy utility")
    ax_ratio.set_xlabel("Number of candidate courses (n)")
    ax_ratio.set_ylabel("Conflict-aware / unconstrained utility")
    ax_ratio.set_ylim(0.5, 1.03)
    ax_ratio.set_xticks(course_counts)
    ax_ratio.legend(frameon=False, fontsize=8)

    for p, color in zip(probabilities, [PALETTE["blue"], PALETTE["green"], PALETTE["vermillion"]]):
        means = [mean(selected_samples[(p, n)]) for n in course_counts]
        errors = [mean_ci(selected_samples[(p, n)])[1] for n in course_counts]
        ax_selected.errorbar(
            course_counts,
            means,
            yerr=errors,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=f"p={p:.0%}",
        )
    ax_selected.set_title("C  Monte Carlo: feasible courses retained")
    ax_selected.set_xlabel("Number of candidate courses (n)")
    ax_selected.set_ylabel("Mean selected courses")
    ax_selected.set_xticks(course_counts)
    ax_selected.legend(frameon=False, fontsize=8)

    names = list(topology)
    losses = [topology[name] for name in names]
    bars = ax_topology.bar(names, losses, color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["vermillion"]])
    ax_topology.set_title("D  Conflict topology matters")
    ax_topology.set_ylabel("Proxy utility loss")
    ax_topology.set_ylim(0, 0.8)
    ax_topology.tick_params(axis="x", rotation=20)
    for bar, loss in zip(bars, losses):
        ax_topology.text(bar.get_x() + bar.get_width() / 2, loss + 0.02, f"{loss:.0%}", ha="center", va="bottom", fontsize=8)

    normalized = Normalize(vmin=0, vmax=max(value for row in matrix for value in row if value == value))
    image = ax_heatmap.imshow(matrix, cmap="PuBu", norm=normalized)
    ax_heatmap.set_title("E  Loss from one pairwise conflict")
    ax_heatmap.set_xticks(range(len(ids)), ids, rotation=35, ha="right")
    ax_heatmap.set_yticks(range(len(ids)), ids)
    for left in range(len(ids)):
        for right in range(len(ids)):
            if matrix[left][right] == matrix[left][right]:
                ax_heatmap.text(right, left, f"{matrix[left][right]:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.04, label="Proxy utility loss")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "conflict_sensitivity.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "conflict_sensitivity.pdf",
        facecolor="white",
        metadata={
            "Creator": "Course Weight Optimizer conflict sensitivity analysis",
            "Title": "Course conflict sensitivity analysis",
            "CreationDate": datetime(2020, 1, 1, tzinfo=timezone.utc),
        },
    )
    plt.close(fig)

    print(f"baseline_proxy_utility={baseline[1]:.4f}")
    for p, invalid, ratio in zip(sample_probabilities, sample_invalid, sample_ratio):
        print(f"sample p={p:.2f} invalid={invalid:.4f} retained_utility={ratio:.4f}")
    print("topology_losses=" + ", ".join(f"{name}:{loss:.4f}" for name, loss in topology.items()))
    print(f"wrote {output_dir / 'conflict_sensitivity.png'}")
    print(f"wrote {output_dir / 'conflict_sensitivity.pdf'}")


if __name__ == "__main__":
    draw()
