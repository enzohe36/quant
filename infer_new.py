"""
Single-step inference for trained PPO stock trading model.

Usage:
  python infer_new.py --checkpoint path/to/model_best.pt \
                      --peer_info path/to/peer_info.pt \
                      --data path/to/feats_new.csv \
                      [--positions current_positions.csv]

Input data must have at least `lookback` rows per symbol.
Positions are inferred for the latest date in the data.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from train_model import Config, PolicyNetwork, load_stock_data, select_greedy, _log


class InferConfig:
    checkpoint: str = "model_best.pt"
    peer_info: str = "peer_info.pt"
    data: str = "feats_new.csv"
    positions: str = ""


def main():
    icfg = InferConfig()
    parser = argparse.ArgumentParser(description="Model inference on new data")

    for name, ann_type in InferConfig.__annotations__.items():
        default = getattr(icfg, name)
        flag = f"--{name}"
        parser.add_argument(flag, type=ann_type, default=default,
                            help=f"(default: {default})")

    args = parser.parse_args()
    for name in InferConfig.__annotations__:
        setattr(icfg, name, getattr(args, name))

    # 1. Load checkpoint
    _log()
    _log("[Checkpoint]")
    ckpt = torch.load(icfg.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = ckpt["n_features"]
    feature_cols = ckpt["feature_cols"]

    # feat_mean/feat_std are registered buffers, restored by load_state_dict
    model = PolicyNetwork(
        n_features, cfg,
        np.zeros(n_features, dtype=np.float32),
        np.ones(n_features, dtype=np.float32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(cfg.device)
    model.eval()

    _log(f"    {'Model':<20s}: {icfg.checkpoint}")
    _log(f"    {'Epoch':<20s}: {ckpt['epoch']}")
    _log(f"    {'Best score':<20s}: {ckpt['best_score']:.4f}")
    _log(f"    {'Parameters':<20s}: {sum(p.numel() for p in model.parameters()):,}")
    del ckpt

    # 2. Load peer info
    _log()
    _log("[Peer Info]")
    peer_info = torch.load(icfg.peer_info, map_location="cpu", weights_only=False)

    K = peer_info["n_peers"]
    sym_to_idx = peer_info["symbol_to_idx"]
    peer_symbols = peer_info["symbols"]
    top_k_idx_all = peer_info["top_k_idx"]    # (N, K)
    top_k_sim_all = peer_info["top_k_sim"]    # (N, K)

    _log(f"    {'Snapshots':<20s}: {peer_info['n_snapshots']}")
    _log(f"    {'Last snapshot':<20s}: {peer_info['last_date']}")
    _log(f"    {'Symbols in map':<20s}: {len(peer_symbols)}")
    del peer_info

    # 3. Load new data
    _log()
    _log("[Data]")
    new_data, new_feature_cols = load_stock_data(icfg.data)

    if new_feature_cols != feature_cols:
        raise ValueError(
            f"Feature mismatch: checkpoint has {len(feature_cols)} features, "
            f"data has {len(new_feature_cols)}"
        )

    target_date = max(d for sd in new_data.values() for d in sd["dates"])
    _log(f"    {'Symbols':<20s}: {len(new_data)}")
    _log(f"    {'Target date':<20s}: {target_date}")

    # 4. Load current positions
    current_positions = {}
    if icfg.positions:
        pos_df = pd.read_csv(icfg.positions, dtype={"symbol": str})
        current_positions = dict(zip(pos_df["symbol"], pos_df["position"]))
        _log(f"    {'Positions loaded':<20s}: {len(current_positions)}")

    # 5. Build observations — self at index 0 + top peers
    K_sub = cfg.train_peers
    symbols_out = []
    obs_list = []
    sim_list = []
    pos_list = []

    for sym, sd in new_data.items():
        if len(sd["stock_features"]) < cfg.lookback:
            continue
        if sym not in sym_to_idx:
            continue

        si = sym_to_idx[sym]
        peer_idx = top_k_idx_all[si]
        peer_sim = top_k_sim_all[si]

        # Select top K_sub peers by similarity
        top_order = np.argsort(-peer_sim)[:K_sub]
        peer_idx = peer_idx[top_order]
        peer_sim = peer_sim[top_order]

        # Self-stock features (slot 0)
        stock_feat = sd["stock_features"][-cfg.lookback:]
        if cfg.pool_self:
            stock_feat = np.zeros_like(stock_feat)

        # Peer features (slots 1..K_sub)
        peer_features = np.zeros(
            (K_sub, cfg.lookback, n_features), dtype=np.float32,
        )
        sim_scores = np.zeros(K_sub + 1, dtype=np.float32)
        sim_scores[0] = 1.0

        for k in range(K_sub):
            pidx = int(peer_idx[k])
            peer_sym = peer_symbols[pidx]
            p_data = new_data.get(peer_sym)
            if p_data is None or len(p_data["stock_features"]) < cfg.lookback:
                continue
            peer_features[k] = p_data["stock_features"][-cfg.lookback:]
            sim_scores[k + 1] = float(peer_sim[k])

        obs = np.concatenate([stock_feat[None], peer_features], axis=0)
        symbols_out.append(sym)
        obs_list.append(obs)
        sim_list.append(sim_scores)
        pos_list.append(current_positions.get(sym, 0.0))

    if not symbols_out:
        _log("    No eligible stocks for inference.")
        return

    _log(f"    {'Eligible stocks':<20s}: {len(symbols_out)}")

    # 6. Batch inference
    _log()
    _log("[Inference]")

    obs_batch = np.stack(obs_list).astype(np.float32)
    sim_batch = np.stack(sim_list).astype(np.float32)
    pos_batch = np.array(pos_list, dtype=np.float32)

    n = len(symbols_out)
    actions = np.empty(n, dtype=np.float32)
    bs = cfg.inference_batch

    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            s = torch.from_numpy(obs_batch[i:j]).to(cfg.device)
            p = torch.from_numpy(pos_batch[i:j]).to(cfg.device)
            c = torch.from_numpy(sim_batch[i:j]).to(cfg.device)
            a, _ = select_greedy(model, s, p, c)
            actions[i:j] = a.cpu().numpy()

    _log(f"    {'Stocks':<20s}: {n}")
    _log(f"    {'Position mean':<20s}: {actions.mean():.4f}")
    _log(f"    {'Position std':<20s}: {actions.std():.4f}")

    # 7. Output
    result = pd.DataFrame({"symbol": symbols_out, "position": actions})
    result.sort_values("symbol", inplace=True)
    result.to_csv("positions.csv", index=False)

    _log(f"    {'Saved file':<20s}: positions.csv")


if __name__ == "__main__":
    main()
