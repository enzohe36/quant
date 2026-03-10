"""
Single-step inference for trained PPO stock trading model.

Usage:
  python infer_new.py --checkpoint path/to/model_best.pt \
                      --peer_info path/to/peer_info.pt \
                      --data path/to/feats_new.csv \
                      [--output positions.csv] \
                      [--positions current_positions.csv]

Input data must have at least `lookback` rows per symbol.
Positions are inferred for the latest date in the data.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from train_model import Config, PolicyNetwork, load_stock_data, select_greedy, _log


def main():
    parser = argparse.ArgumentParser(description="Model inference on new data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--peer_info", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="positions.csv")
    parser.add_argument("--positions", default=None,
                        help="Current positions CSV (symbol,position)")
    args = parser.parse_args()

    # 1. Load checkpoint
    _log()
    _log("[Checkpoint]")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    _log(f"    {'Model':<20s}: {args.checkpoint}")
    _log(f"    {'Epoch':<20s}: {ckpt['epoch']}")
    _log(f"    {'Best score':<20s}: {ckpt['best_score']:.4f}")
    _log(f"    {'Parameters':<20s}: {sum(p.numel() for p in model.parameters()):,}")
    del ckpt

    # 2. Load peer info
    _log()
    _log("[Peer Info]")
    peer_info = torch.load(args.peer_info, map_location="cpu", weights_only=False)

    K = peer_info["n_peers"]
    sym_to_idx = peer_info["symbol_to_idx"]
    peer_symbols = peer_info["symbols"]

    last_col = int(peer_info["checkpoint_cols"][-1])
    last_date = peer_info["all_dates"][last_col]
    top_k_idx_all = peer_info["all_top_k_idx"][-1]   # (N, K)
    top_k_corr_all = peer_info["all_top_k_corr"][-1]  # (N, K)

    _log(f"    {'Peer snapshots':<20s}: {len(peer_info['checkpoint_cols'])}")
    _log(f"    {'Last snapshot':<20s}: {last_date}")
    _log(f"    {'Symbols in map':<20s}: {len(peer_symbols)}")
    del peer_info

    # 3. Load new data
    _log()
    _log("[Data]")
    new_data, new_feature_cols = load_stock_data(args.data)

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
    if args.positions:
        pos_df = pd.read_csv(args.positions, dtype={"symbol": str})
        current_positions = dict(zip(pos_df["symbol"], pos_df["position"]))
        _log(f"    {'Positions loaded':<20s}: {len(current_positions)}")

    # 5. Build observations
    symbols_out = []
    obs_list = []
    corr_list = []
    mask_list = []
    pos_list = []

    for sym, sd in new_data.items():
        if len(sd["stock_features"]) < cfg.lookback:
            continue
        if sym not in sym_to_idx:
            continue

        stock_feat = sd["stock_features"][-cfg.lookback:]
        si = sym_to_idx[sym]
        peer_idx = top_k_idx_all[si]
        peer_corr = top_k_corr_all[si]

        peer_features = np.zeros(
            (K, cfg.lookback, n_features), dtype=np.float32,
        )
        corr_scores = np.zeros(K + 1, dtype=np.float32)
        peer_mask = np.ones(K + 1, dtype=bool)

        corr_scores[0] = 1.0
        peer_mask[0] = False

        for k in range(K):
            pidx = int(peer_idx[k])
            pcorr = float(peer_corr[k])
            if pcorr <= 0:
                continue
            peer_sym = peer_symbols[pidx]
            p_data = new_data.get(peer_sym)
            if p_data is None or len(p_data["stock_features"]) < cfg.lookback:
                continue
            peer_features[k] = p_data["stock_features"][-cfg.lookback:]
            corr_scores[k + 1] = pcorr
            peer_mask[k + 1] = False

        obs = np.concatenate(
            [stock_feat[None], peer_features], axis=0,
        )
        symbols_out.append(sym)
        obs_list.append(obs)
        corr_list.append(corr_scores)
        mask_list.append(peer_mask)
        pos_list.append(current_positions.get(sym, 0.0))

    if not symbols_out:
        _log("    No eligible stocks for inference.")
        return

    _log(f"    {'Eligible stocks':<20s}: {len(symbols_out)}")

    # 6. Batch inference
    _log()
    _log("[Inference]")

    obs_batch = np.stack(obs_list).astype(np.float32)
    corr_batch = np.stack(corr_list).astype(np.float32)
    mask_batch = np.stack(mask_list)
    pos_batch = np.array(pos_list, dtype=np.float32)

    n = len(symbols_out)
    actions = np.empty(n, dtype=np.float32)
    bs = cfg.inference_batch_size

    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            s = torch.from_numpy(obs_batch[i:j]).to(cfg.device)
            p = torch.from_numpy(pos_batch[i:j]).to(cfg.device)
            c = torch.from_numpy(corr_batch[i:j]).to(cfg.device)
            m = torch.from_numpy(mask_batch[i:j]).to(cfg.device)
            a, _ = select_greedy(model, s, p, c, m)
            actions[i:j] = a.cpu().numpy()

    _log(f"    {'Stocks':<20s}: {n}")
    _log(f"    {'Position mean':<20s}: {actions.mean():.4f}")
    _log(f"    {'Position std':<20s}: {actions.std():.4f}")

    # 7. Output
    result = pd.DataFrame({"symbol": symbols_out, "position": actions})
    result.sort_values("symbol", inplace=True)
    result.to_csv(args.output, index=False)

    _log(f"    {'Saved file':<20s}: {args.output}")


if __name__ == "__main__":
    main()
