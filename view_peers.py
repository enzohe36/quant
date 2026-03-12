#!/usr/bin/env python3
"""Visualize peer_info.pt: similarity distributions, adjacency matrix, rank decay."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import torch


class Config:
    peer_info: str = "checkpoints/peer_info.pt"


def load_peer_info(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_adjacency(adj, date_label, out_dir):
    """Plot a single adjacency matrix with dendrograms (view_peers_ref style)."""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, adj.shape[0] * 10))

    N = adj.shape[0]
    adj = adj.copy()
    np.fill_diagonal(adj, 0.0)

    dist_mat = np.maximum(1.0 - adj, 0.0)
    np.fill_diagonal(dist_mat, 0.0)
    condensed = dist_mat[np.triu_indices(N, k=1)]
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    adj_sorted = adj[np.ix_(order, order)]

    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 5, 0.2],
                          height_ratios=[1, 5],
                          wspace=0.01, hspace=0.01)

    ax_top = fig.add_subplot(gs[0, 1])
    dendrogram(Z, ax=ax_top, no_labels=True, color_threshold=0,
               above_threshold_color="steelblue")
    for coll in ax_top.collections:
        coll.set_linewidths(0.1)
    ax_top.axis("off")
    ax_top.set_title(f"Peer Adjacency — {date_label}")

    ax_left = fig.add_subplot(gs[1, 0])
    dendrogram(Z, ax=ax_left, no_labels=True, color_threshold=0,
               above_threshold_color="steelblue", orientation="left")
    for coll in ax_left.collections:
        coll.set_linewidths(0.1)
    ax_left.axis("off")

    ax_adj = fig.add_subplot(gs[1, 1])
    im = ax_adj.imshow(adj_sorted, cmap="RdBu_r", aspect="auto",
                       interpolation="nearest", vmin=-1, vmax=1)
    ax_adj.set_xticks([])
    ax_adj.set_yticks([])

    ax_cb = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cb, label="Correlation")

    date_str = str(date_label).replace("-", "")
    adj_path = os.path.join(out_dir, f"adjacency_{date_str}.png")
    fig.savefig(adj_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sys.setrecursionlimit(old_limit)
    print(f"Saved {adj_path}")


def plot_all(peer_info, cfg):
    sim = peer_info["all_top_k_sim"]       # (n_cp, N, K)
    snapshot_cols = peer_info["snapshot_cols"]
    all_dates = peer_info["all_dates"]
    _, _, K = sim.shape
    out_dir = os.path.dirname(os.path.abspath(cfg.peer_info))

    dates = [all_dates[c] if c < len(all_dates) else c for c in snapshot_cols]

    # Adjacency heatmaps — separate file per diagnostic snapshot
    sim_diagnostics = peer_info["sim_diagnostics"]       # list of 3 (N, N) arrays
    sim_diagnostic_cols = peer_info["sim_diagnostic_cols"]  # list of 3 snapshot indices
    for snap, cp in zip(sim_diagnostics, sim_diagnostic_cols):
        col = snapshot_cols[cp]
        date_label = all_dates[col] if col < len(all_dates) else col
        plot_adjacency(snap, date_label, out_dir)

    # Similarity diagnostics — single figure (train_model style)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Mean similarity over time (valid stocks only)
    first_col = peer_info["global_first_col"]   # (N,)
    last_col = peer_info["global_last_col"]      # (N,)
    cp_cols = snapshot_cols                     # (n_cp,)
    valid = (first_col[None, :] <= cp_cols[:, None]) & \
            (cp_cols[:, None] <= last_col[None, :])   # (n_cp, N)
    valid_sim = np.where(valid[:, :, None], sim, np.nan)
    mean_sim = np.nanmean(valid_sim, axis=(1, 2))  # (n_cp,)
    has_positive = (sim > 0).any(axis=2)       # (n_cp, N)
    axes[0].plot(dates, mean_sim, linewidth=1.0)
    axes[0].set_title("Top-K Similarity")
    axes[0].grid(True, alpha=0.3)

    # Zero-peer stocks over time (only stocks valid at each snapshot)
    zero_count = (valid & ~has_positive).sum(axis=1)   # (n_cp,)
    axes[1].plot(dates, zero_count, linewidth=1.0)
    axes[1].set_title("Stocks Without Peers")
    axes[1].grid(True, alpha=0.3)

    # Rank decay
    mean_by_rank = np.nanmean(valid_sim, axis=(0, 1))  # (K,)
    axes[2].plot(range(K), mean_by_rank, marker="o", color="steelblue")
    axes[2].set_title("Similarity by Peer Rank")
    axes[2].set_xticks(range(K))
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in axes[:2]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    diag_path = os.path.join(out_dir, "peer_plots.png")
    fig.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {diag_path}")


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Visualize peer_info.pt")

    for name, ann_type in Config.__annotations__.items():
        default = getattr(cfg, name)
        flag = f"--{name}"
        parser.add_argument(flag, type=ann_type, default=default,
                            help=f"(default: {default})")

    args = parser.parse_args()
    for name in Config.__annotations__:
        setattr(cfg, name, getattr(args, name))

    peer_info = load_peer_info(cfg.peer_info)

    last_col = peer_info["snapshot_cols"][-1]
    all_dates = peer_info["all_dates"]
    date_label = all_dates[last_col] if last_col < len(all_dates) else last_col
    print(f"Using last snapshot (col {last_col}, date {date_label})")

    plot_all(peer_info, cfg)


if __name__ == "__main__":
    main()
