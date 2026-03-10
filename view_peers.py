#!/usr/bin/env python3
"""Visualize peer_info.pt: correlation distributions, adjacency matrix, rank decay."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage


def load_peer_info(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def find_nearest_checkpoint(peer_info, date_str):
    """Return the checkpoint index closest to the given date string."""
    date_to_col = peer_info["date_to_col"]
    checkpoint_cols = peer_info["checkpoint_cols"]

    # Match date key type (could be str or datetime-like)
    sample_key = next(iter(date_to_col))
    if isinstance(sample_key, str):
        target = date_str
    else:
        import datetime
        target = datetime.date.fromisoformat(date_str)

    if target not in date_to_col:
        # Find the closest date that exists
        all_dates = sorted(date_to_col.keys())
        closest = min(all_dates, key=lambda d: abs(
            (d if isinstance(d, int) else
             (d.toordinal() if hasattr(d, "toordinal") else 0))
            - (target if isinstance(target, int) else
               (target.toordinal() if hasattr(target, "toordinal") else 0))
        ))
        print(f"Date {date_str} not found; using closest: {closest}")
        target = closest

    col = date_to_col[target]
    cp_idx = int(np.searchsorted(checkpoint_cols, col))
    cp_idx = min(cp_idx, len(checkpoint_cols) - 1)
    # Check if the previous checkpoint is actually closer
    if cp_idx > 0:
        if abs(checkpoint_cols[cp_idx - 1] - col) < abs(checkpoint_cols[cp_idx] - col):
            cp_idx -= 1
    return cp_idx


def plot_corr_distributions(peer_info, out_dir):
    """Plot #1: correlation distribution at first / middle / last checkpoint."""
    corr = peer_info["all_top_k_corr"]  # (n_cp, N, K)
    n_cp = corr.shape[0]
    indices = [0, n_cp // 2, n_cp - 1]
    checkpoint_cols = peer_info["checkpoint_cols"]
    all_dates = peer_info["all_dates"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, cp_idx in zip(axes, indices):
        vals = corr[cp_idx].ravel()
        ax.hist(vals, bins=50, edgecolor="black", linewidth=0.3)
        col = checkpoint_cols[cp_idx]
        date_label = all_dates[col] if col < len(all_dates) else col
        ax.set_title(f"Checkpoint {cp_idx}\n{date_label}")
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Count")
    fig.suptitle("Top-K Peer Correlation Distribution", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "peer_corr_dist.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_adjacency(peer_info, cp_idx, out_dir):
    """Plot #3: adjacency matrix at a given checkpoint, clustered."""
    N = len(peer_info["symbols"])
    K = peer_info["n_peers"]
    idx = peer_info["all_top_k_idx"][cp_idx]    # (N, K)
    corr = peer_info["all_top_k_corr"][cp_idx]  # (N, K)

    # Build symmetric adjacency: average of (i->j) and (j->i) weights
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            j = idx[i, k]
            adj[i, j] = max(adj[i, j], corr[i, k])
    adj = np.maximum(adj, adj.T)

    # Hierarchical clustering for row/col reordering
    dist = 1.0 - adj
    np.fill_diagonal(dist, 0.0)
    condensed = dist[np.triu_indices(N, k=1)]
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]

    adj_sorted = adj[np.ix_(order, order)]

    col = peer_info["checkpoint_cols"][cp_idx]
    all_dates = peer_info["all_dates"]
    date_label = all_dates[col] if col < len(all_dates) else col

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(adj_sorted, cmap="hot", aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    ax.set_title(f"Peer Adjacency Matrix (clustered)\n{date_label}", fontweight="bold")
    ax.set_xlabel("Stock (clustered order)")
    ax.set_ylabel("Stock (clustered order)")
    plt.tight_layout()
    path = os.path.join(out_dir, "peer_adjacency.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_rank_decay(peer_info, out_dir):
    """Plot #4: mean correlation by peer rank."""
    corr = peer_info["all_top_k_corr"]  # (n_cp, N, K)
    mean_by_rank = corr.mean(axis=(0, 1))  # (K,)
    K = len(mean_by_rank)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(K), mean_by_rank, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Peer rank")
    ax.set_ylabel("Mean correlation")
    ax.set_title("Average Correlation by Peer Rank", fontweight="bold")
    ax.set_xticks(range(0, K, max(1, K // 10)))
    plt.tight_layout()
    path = os.path.join(out_dir, "peer_rank_decay.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize peer_info.pt")
    parser.add_argument("--peer-info", required=True, help="Path to peer_info.pt")
    parser.add_argument("--date", required=True, help="Snapshot date for adjacency matrix (YYYY-MM-DD)")
    args = parser.parse_args()

    peer_info = load_peer_info(args.peer_info)
    out_dir = os.path.dirname(os.path.abspath(args.peer_info))

    cp_idx = find_nearest_checkpoint(peer_info, args.date)
    col = peer_info["checkpoint_cols"][cp_idx]
    all_dates = peer_info["all_dates"]
    date_label = all_dates[col] if col < len(all_dates) else col
    print(f"Using checkpoint {cp_idx} (col {col}, date {date_label})")

    plot_corr_distributions(peer_info, out_dir)
    plot_adjacency(peer_info, cp_idx, out_dir)
    plot_rank_decay(peer_info, out_dir)
    plt.show()


if __name__ == "__main__":
    main()
