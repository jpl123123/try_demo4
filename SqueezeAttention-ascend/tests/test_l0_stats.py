"""L0: squeeze statistics (kmeans, budgets, windows) - pure numpy."""

import numpy as np

from squeeze_ascend.stats import (
    budgets_to_windows,
    compute_layer_budgets,
    kmeans_1d,
)


def test_kmeans_separates_three_groups():
    means = np.concatenate([np.full(24, 0.95), np.full(24, 0.5), np.full(24, 0.05)])
    labels = kmeans_1d(means, k=3)
    assert len(set(labels.tolist())) == 3
    counts = np.bincount(labels)
    assert counts.min() == 24 and counts.max() == 24


def test_budget_total_conserved():
    means = np.concatenate([np.full(24, 0.95), np.full(24, 0.5), np.full(24, 0.05)])
    budgets = compute_layer_budgets(means, ini_size=0.21, kv_class3=0.08, clusters=3)
    assert abs(budgets.sum() - 72 * 0.21) < 1e-6
    assert np.allclose(budgets[means > 0.9], 0.08)  # class3 = high-importance


def test_uniform_mode():
    means = np.concatenate([np.full(24, 0.95), np.full(24, 0.5), np.full(24, 0.05)])
    u = compute_layer_budgets(means, ini_size=0.21, kv_class3=0.21, clusters=3)
    assert np.allclose(u, 0.21)


def test_degenerate_equal_means_falls_back_uniform():
    e = compute_layer_budgets(np.full(48, 0.3), 0.2, 0.05, 3)
    assert np.allclose(e, 0.2)
    e2 = compute_layer_budgets(np.full(48, 0.3) + np.linspace(0, 1e-10, 48), 0.2, 0.05, 3)
    assert abs(e2.sum() - 48 * 0.2) < 1e-6


def test_windows_respect_bounds_and_total():
    means = np.concatenate([np.full(24, 0.95), np.full(24, 0.5), np.full(24, 0.05)])
    budgets = compute_layer_budgets(means, ini_size=0.21, kv_class3=0.08, clusters=3)
    w = budgets_to_windows(budgets, 10000, 4)
    assert (w >= 4).all() and (w <= 10000).all()
    assert abs(w.sum() - 72 * 0.21 * 10000) <= 72
    # start_size clamp
    w2 = budgets_to_windows(np.array([0.002]), 1000, 4)
    assert w2[0] == 4
