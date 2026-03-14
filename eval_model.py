"""
Diagnose a trained checkpoint: feature attribution, representation analysis,
attention patterns, and test-set evaluation with ablation.

Usage:
  python diagnose_model.py --checkpoint checkpoints/model_best.pt \
                           --data feats.csv \
                           [--n_episodes 500] [--batch 256]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn.functional as F

from train_model import (
    Config, PolicyNetwork, _log, _record_eval, _EVAL_METRICS,
    load_stock_data, _split_stock_data, build_global_features,
    compute_peer_maps, _peer_map_worker, compute_feature_stats,
    generate_episodes, evaluate_episodes, evaluate_ablated,
    compute_episode_returns, compute_baseline_rewards,
    VecEnv, select_greedy,
)


class DiagConfig:
    checkpoint: str = "checkpoints/model_best.pt"
    data: str = "feats.csv"
    n_episodes: int = 500
    batch: int = 256
    seed: int = 42


def _load_model_and_data(dcfg):
    """Load checkpoint, rebuild data splits and peer info."""
    _log()
    _log("[Checkpoint]")
    ckpt = torch.load(dcfg.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config()
    for k, v in ckpt["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.data_path = dcfg.data

    n_features = ckpt["n_features"]
    feature_cols = ckpt["feature_cols"]

    model = PolicyNetwork(
        n_features, cfg,
        np.zeros(n_features, dtype=np.float32),
        np.ones(n_features, dtype=np.float32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(cfg.device)
    model.eval()

    _log(f"    {'Model':<20s}: {dcfg.checkpoint}")
    _log(f"    {'Epoch':<20s}: {ckpt['epoch']}")
    _log(f"    {'Best score':<20s}: {ckpt['best_score']:.4f}")
    del ckpt

    _log()
    _log("[Data]")
    all_data, new_feature_cols = load_stock_data(cfg.data_path)
    assert new_feature_cols == feature_cols, "Feature mismatch"
    train_data, val_data, test_data = _split_stock_data(
        all_data, cfg.train_ratio, cfg.val_ratio,
    )

    import tempfile, torch.multiprocessing as mp
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_peer_map_worker,
                    args=(all_data, feature_cols, cfg, tmp.name))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Peer map subprocess failed (exit code {p.exitcode})")
    peer_info = torch.load(tmp.name, map_location="cpu", weights_only=False)
    os.unlink(tmp.name)

    gf, gf_first_col, gf_last_col, gf_delist = build_global_features(
        all_data, peer_info["symbols"], peer_info["symbol_to_idx"],
        peer_info["date_to_col"], len(peer_info["all_dates"]),
    )
    peer_info["global_features"] = gf
    peer_info["global_first_col"] = gf_first_col
    peer_info["global_last_col"] = gf_last_col
    peer_info["global_delist"] = gf_delist

    _log(f"    {'Symbols':<20s}: {len(all_data)}")
    _log(f"    {'Train':<20s}: {len(train_data)}")
    _log(f"    {'Val':<20s}: {len(val_data)}")
    _log(f"    {'Test':<20s}: {len(test_data)}")

    return model, cfg, feature_cols, peer_info, train_data, val_data, test_data


def _generate_split_episodes(split_data, peer_info, cfg, seed_offset, n_max):
    rng = np.random.default_rng(cfg.seed + seed_offset)
    episodes = generate_episodes(split_data, peer_info, cfg, rng, is_train=False)
    if len(episodes) > n_max:
        np.random.default_rng(cfg.seed).shuffle(episodes)
        episodes = episodes[:n_max]
    return episodes


def _build_obs_batch(episodes, cfg, global_features, max_n=None):
    """Build observation tensors from episodes for diagnostic analysis.

    Reconstructs observations the same way VecEnv does: self-stock features
    at index 0 (from episode stock_features), peers from global_features.
    """
    if max_n and len(episodes) > max_n:
        episodes = episodes[:max_n]

    all_obs, all_sim, all_pos = [], [], []
    for ep in episodes:
        K = ep["step_peer_map"].shape[1]
        T = cfg.lookback
        # Use the middle step of each episode
        step = cfg.episode_length // 2
        # Self-stock features (slot 0)
        self_window = ep["stock_features"][step : step + T][None, :, :]  # (1, T, F)
        # Peer features from global_features
        date_cols = ep["date_cols"]
        col_window = date_cols[step : step + T]
        step_idx = ep["step_peer_map"][step]  # (K,)
        sym_exp = step_idx[:, None]  # (K, 1)
        col_exp = col_window[None, :]  # (1, T)
        peer_window = global_features[sym_exp, col_exp]  # (K, T, F)
        obs = np.concatenate([self_window, peer_window], axis=0).astype(np.float32)
        sim = ep["sim_scores"][step]  # (K+1,)
        all_obs.append(obs)
        all_sim.append(sim)
        all_pos.append(0.0)

    return (
        np.stack(all_obs),
        np.stack(all_sim),
        np.array(all_pos, dtype=np.float32),
    )


# 1. Feature Attribution =======================================================

def diagnose_feature_attribution(model, train_episodes, val_episodes, cfg,
                                 global_features, feature_cols, out_dir):
    """Gradient-based feature attribution: which features drive policy output."""
    _log()
    _log("[Feature Attribution]")

    results = {}
    for name, episodes in [("train", train_episodes), ("val", val_episodes)]:
        obs_np, sim_np, pos_np = _build_obs_batch(episodes, cfg, global_features)
        obs = torch.from_numpy(obs_np).to(cfg.device).requires_grad_(True)
        sim = torch.from_numpy(sim_np).to(cfg.device)
        pos = torch.from_numpy(pos_np).to(cfg.device)

        mu, _, _ = model(obs, pos, sim)
        mu.sum().backward()

        # Mean absolute gradient per feature (averaged over batch, stocks, time)
        grad = obs.grad.abs().mean(dim=(0, 1, 2)).cpu().numpy()  # (F,)
        results[name] = grad
        _log(f"    {name + ' top-5 features':<20s}: {[feature_cols[i] for i in np.argsort(-grad)[:5]]}")

    train_grad = results["train"]
    val_grad = results["val"]

    # Features that are important for train but not val = memorization candidates
    ratio = (train_grad + 1e-10) / (val_grad + 1e-10)
    memorization_idx = np.argsort(-ratio)[:10]
    _log(f"    {'Overfit features':<20s}: {[feature_cols[i] for i in memorization_idx]}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    idx_sorted = np.argsort(-train_grad)[:20]
    labels = [feature_cols[i] for i in idx_sorted]
    x = np.arange(len(labels))
    axes[0].barh(x, train_grad[idx_sorted], alpha=0.7, label="Train")
    axes[0].barh(x, val_grad[idx_sorted], alpha=0.7, label="Val")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_title("Feature Attribution (top 20 by train)")
    axes[0].legend(fontsize=7)

    axes[1].scatter(train_grad, val_grad, alpha=0.5, s=10)
    axes[1].set_xlabel("Train attribution")
    axes[1].set_ylabel("Val attribution")
    axes[1].set_title("Train vs Val Attribution")
    axes[1].plot([0, train_grad.max()], [0, train_grad.max()],
                 "k--", alpha=0.3, label="y=x")
    axes[1].legend(fontsize=7)

    axes[2].barh(range(10), ratio[memorization_idx[:10]])
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels([feature_cols[i] for i in memorization_idx[:10]], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_title("Train/Val Attribution Ratio\n(memorization candidates)")

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_attribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"    {'Saved file':<20s}: {path}")


# 2. Representation Analysis ===================================================

def diagnose_representations(model, train_episodes, val_episodes, cfg,
                             global_features, out_dir):
    """Compare hidden representations between train and val."""
    _log()
    _log("[Representation Analysis]")

    def _extract_repr(episodes):
        obs_np, sim_np, pos_np = _build_obs_batch(episodes, cfg, global_features)
        obs = torch.from_numpy(obs_np).to(cfg.device)
        sim = torch.from_numpy(sim_np).to(cfg.device)

        with torch.no_grad():
            B, K, T, Ft = obs.shape
            x = (obs - model.feat_mean) / model.feat_std
            hidden = model.input_norm(model.input_proj(x.reshape(B, K * T, Ft)))
            hidden = hidden + model.pos_emb.repeat(1, K, 1)
            sim_exp = sim.unsqueeze(2).expand(B, K, T).reshape(B, K * T, 1)
            hidden = hidden + model.sim_proj(sim_exp)
            hidden = model.transformer(hidden)
            pooled = hidden.mean(dim=1)  # (B, d_model)
        return pooled.cpu().numpy()

    train_repr = _extract_repr(train_episodes)
    val_repr = _extract_repr(val_episodes)

    # Statistics
    train_norms = np.linalg.norm(train_repr, axis=1)
    val_norms = np.linalg.norm(val_repr, axis=1)
    _log(f"    {'Train repr norm':<20s}: {train_norms.mean():.4f}")
    _log(f"    {'Train repr std':<20s}: {train_norms.std():.4f}")
    _log(f"    {'Val repr norm':<20s}: {val_norms.mean():.4f}")
    _log(f"    {'Val repr std':<20s}: {val_norms.std():.4f}")

    # Intra-split cosine similarity
    def _mean_cosine(X):
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        n = min(500, len(X_norm))
        X_sub = X_norm[:n]
        sim_mat = X_sub @ X_sub.T
        mask = np.triu_indices(n, k=1)
        return float(sim_mat[mask].mean())

    train_cos = _mean_cosine(train_repr)
    val_cos = _mean_cosine(val_repr)
    _log(f"    {'Train intra cosine':<20s}: {train_cos:.4f}")
    _log(f"    {'Val intra cosine':<20s}: {val_cos:.4f}")

    # Cross-split cosine similarity
    t_norm = train_repr / (np.linalg.norm(train_repr, axis=1, keepdims=True) + 1e-8)
    v_norm = val_repr / (np.linalg.norm(val_repr, axis=1, keepdims=True) + 1e-8)
    n_t, n_v = min(500, len(t_norm)), min(500, len(v_norm))
    cross_cos = float((t_norm[:n_t] @ v_norm[:n_v].T).mean())
    _log(f"    {'Cross cosine':<20s}: {cross_cos:.4f}")

    # PCA visualization
    combined = np.concatenate([train_repr, val_repr])
    mean = combined.mean(axis=0)
    centered = combined - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T
    n_train = len(train_repr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(proj[:n_train, 0], proj[:n_train, 1],
                    alpha=0.3, s=5, label="Train", color="b")
    axes[0].scatter(proj[n_train:, 0], proj[n_train:, 1],
                    alpha=0.3, s=5, label="Val", color="r")
    axes[0].set_title("Representations (PCA)")
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].hist(train_norms, bins=50, alpha=0.6, label="Train", density=True)
    axes[1].hist(val_norms, bins=50, alpha=0.6, label="Val", density=True)
    axes[1].set_title("Representation Norms")
    axes[1].legend(fontsize=7)

    # Per-dimension variance comparison
    train_var = train_repr.var(axis=0)
    val_var = val_repr.var(axis=0)
    dim_idx = np.argsort(-train_var)[:50]
    axes[2].plot(train_var[dim_idx], label="Train", alpha=0.7)
    axes[2].plot(val_var[dim_idx], label="Val", alpha=0.7)
    axes[2].set_title("Per-Dimension Variance (top 50)")
    axes[2].legend(fontsize=7)
    axes[2].set_xlabel("Dimension (sorted by train var)")

    plt.tight_layout()
    path = os.path.join(out_dir, "representations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"    {'Saved file':<20s}: {path}")


# 3. Attention Patterns ========================================================

def diagnose_attention(model, train_episodes, val_episodes, cfg,
                       global_features, out_dir):
    """Extract and compare attention patterns between train and val."""
    _log()
    _log("[Attention Patterns]")

    def _extract_attention(episodes):
        obs_np, sim_np, pos_np = _build_obs_batch(episodes, cfg, global_features)
        obs = torch.from_numpy(obs_np).to(cfg.device)
        sim = torch.from_numpy(sim_np).to(cfg.device)

        with torch.no_grad():
            B, K, T, Ft = obs.shape
            x = (obs - model.feat_mean) / model.feat_std
            hidden = model.input_norm(model.input_proj(x.reshape(B, K * T, Ft)))
            hidden = hidden + model.pos_emb.repeat(1, K, 1)
            sim_exp = sim.unsqueeze(2).expand(B, K, T).reshape(B, K * T, 1)
            hidden = hidden + model.sim_proj(sim_exp)

            # Hook into the last transformer layer to get attention weights
            attn_weights = []

            def hook_fn(module, args, kwargs, output):
                # TransformerEncoderLayer with batch_first=True
                # Recompute attention manually
                src = args[0] if args else kwargs.get("src", None)
                sa = module.self_attn
                q = k = v = src
                attn_out, weights = sa(q, k, v, need_weights=True,
                                       average_attn_weights=True)
                attn_weights.append(weights.cpu())
                return output

            last_layer = model.transformer.layers[-1]
            handle = last_layer.register_forward_hook(hook_fn, with_kwargs=True)
            model.transformer(hidden)
            handle.remove()

            attn = attn_weights[0].numpy()  # (B, K*T, K*T)

        return attn, K, T

    train_attn, K, T = _extract_attention(train_episodes)
    val_attn, _, _ = _extract_attention(val_episodes)

    # Reshape to (B, K, T, K, T) for analysis
    B_train = train_attn.shape[0]
    B_val = val_attn.shape[0]
    train_attn_r = train_attn.reshape(B_train, K, T, K, T)
    val_attn_r = val_attn.reshape(B_val, K, T, K, T)

    # Self-stock to peer attention: how much do self tokens attend to peer tokens?
    # Self is the stock with highest sim. In eval mode (top-K by sim), self is
    # typically at index 0 (sim=1.0 sorts first).
    self_to_self = train_attn_r[:, 0, :, 0, :].mean()  # self attending to self
    self_to_peers = train_attn_r[:, 0, :, 1:, :].mean()  # self attending to peers
    _log(f"    {'Train self->self':<20s}: {self_to_self:.4f}")
    _log(f"    {'Train self->peers':<20s}: {self_to_peers:.4f}")

    self_to_self_v = val_attn_r[:, 0, :, 0, :].mean()
    self_to_peers_v = val_attn_r[:, 0, :, 1:, :].mean()
    _log(f"    {'Val self->self':<20s}: {self_to_self_v:.4f}")
    _log(f"    {'Val self->peers':<20s}: {self_to_peers_v:.4f}")

    # Temporal attention pattern: average over batch and heads, per temporal offset
    # How much does each time step attend to each other time step (within self)?
    train_temporal = train_attn_r[:, 0, :, 0, :].mean(axis=0)  # (T, T)
    val_temporal = val_attn_r[:, 0, :, 0, :].mean(axis=0)  # (T, T)

    # Stock-level attention: collapse temporal dimensions
    train_stock_attn = train_attn_r.mean(axis=(2, 4))  # (B, K, K)
    val_stock_attn = val_attn_r.mean(axis=(2, 4))
    train_stock_avg = train_stock_attn.mean(axis=0)  # (K, K)
    val_stock_avg = val_stock_attn.mean(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Temporal attention (self->self) for train and val
    im0 = axes[0, 0].imshow(train_temporal, aspect="auto", interpolation="nearest")
    axes[0, 0].set_title("Train: Self Temporal Attention")
    axes[0, 0].set_xlabel("Key time step")
    axes[0, 0].set_ylabel("Query time step")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(val_temporal, aspect="auto", interpolation="nearest")
    axes[0, 1].set_title("Val: Self Temporal Attention")
    axes[0, 1].set_xlabel("Key time step")
    axes[0, 1].set_ylabel("Query time step")
    fig.colorbar(im1, ax=axes[0, 1])

    # Temporal attention difference
    im2 = axes[0, 2].imshow(train_temporal - val_temporal, aspect="auto",
                             interpolation="nearest", cmap="RdBu_r")
    axes[0, 2].set_title("Train - Val Temporal Attention")
    axes[0, 2].set_xlabel("Key time step")
    axes[0, 2].set_ylabel("Query time step")
    fig.colorbar(im2, ax=axes[0, 2])

    # Stock-level attention
    im3 = axes[1, 0].imshow(train_stock_avg, aspect="auto", interpolation="nearest")
    axes[1, 0].set_title("Train: Stock Attention")
    axes[1, 0].set_xlabel("Key stock")
    axes[1, 0].set_ylabel("Query stock")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(val_stock_avg, aspect="auto", interpolation="nearest")
    axes[1, 1].set_title("Val: Stock Attention")
    axes[1, 1].set_xlabel("Key stock")
    axes[1, 1].set_ylabel("Query stock")
    fig.colorbar(im4, ax=axes[1, 1])

    # Attention entropy per query position (how spread out is attention?)
    def _entropy(attn_matrix):
        # attn_matrix: (B, seq, seq)
        eps = 1e-8
        log_a = np.log(attn_matrix + eps)
        ent = -(attn_matrix * log_a).sum(axis=-1)  # (B, seq)
        return ent.mean(axis=0)  # (seq,)

    train_ent = _entropy(train_attn)
    val_ent = _entropy(val_attn)
    axes[1, 2].plot(train_ent, alpha=0.7, label="Train")
    axes[1, 2].plot(val_ent, alpha=0.7, label="Val")
    axes[1, 2].set_title("Attention Entropy per Position")
    axes[1, 2].set_xlabel("Token position")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "attention_patterns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"    {'Saved file':<20s}: {path}")


# 4. Test Evaluation with Ablation =============================================

def diagnose_test_eval(model, test_episodes, cfg, global_features, out_dir):
    """Run test evaluation with full model and ablation."""
    _log()
    _log("[Test Evaluation]")
    _log(f"    {'Test episodes':<20s}: {len(test_episodes)}")

    test_result = evaluate_episodes(model, test_episodes, cfg, global_features)
    _log(f"    {'Test baseline':<20s}: {test_result['baseline_return']:.4f}")
    _record_eval(history=None, result=test_result, prefix="test", log_label="Test")

    _log()
    _log("[Test Ablation]")
    ablation = evaluate_ablated(model, test_episodes, cfg, global_features)
    _record_eval(history=None, result=ablation["no_peers"],
                 prefix="no_peers", log_label="No-peers")
    _record_eval(history=None, result=ablation["no_stock"],
                 prefix="no_stock", log_label="No-stock")


def main():
    dcfg = DiagConfig()
    parser = argparse.ArgumentParser(description="Diagnose trained model")

    for name, ann_type in DiagConfig.__annotations__.items():
        default = getattr(dcfg, name)
        flag = f"--{name}"
        parser.add_argument(flag, type=ann_type, default=default,
                            help=f"(default: {default})")

    args, _ = parser.parse_known_args()
    for name in DiagConfig.__annotations__:
        setattr(dcfg, name, getattr(args, name))

    model, cfg, feature_cols, peer_info, train_data, val_data, test_data = \
        _load_model_and_data(dcfg)
    global_features = peer_info["global_features"]

    out_dir = os.path.dirname(os.path.abspath(dcfg.checkpoint))
    n = dcfg.n_episodes

    _log()
    _log("[Generating Episodes]")
    train_episodes = _generate_split_episodes(train_data, peer_info, cfg, 0, n)
    val_episodes = _generate_split_episodes(val_data, peer_info, cfg, 1, n)
    test_episodes = _generate_split_episodes(test_data, peer_info, cfg, 1000, n)
    _log(f"    {'Train episodes':<20s}: {len(train_episodes)}")
    _log(f"    {'Val episodes':<20s}: {len(val_episodes)}")
    _log(f"    {'Test episodes':<20s}: {len(test_episodes)}")

    diagnose_feature_attribution(
        model, train_episodes, val_episodes, cfg, global_features,
        feature_cols, out_dir,
    )
    diagnose_representations(
        model, train_episodes, val_episodes, cfg, global_features, out_dir,
    )
    diagnose_attention(
        model, train_episodes, val_episodes, cfg, global_features, out_dir,
    )
    diagnose_test_eval(model, test_episodes, cfg, global_features, out_dir)

    _log()
    _log("Diagnostics complete.")


if __name__ == "__main__":
    main()
