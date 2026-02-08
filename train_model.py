"""
PPO + Transformer RL for stock trading.

Stock CSV: symbol, date, price, <stock_feat_1>, ...
Market CSV: date, <mkt_feat_1>, ...  (all columns except date start with "mkt_")

Market features are loaded once from a single file and joined to stock features
by date when building episodes. The model receives [stock | market] features.
Features with "mkt_" prefix route through the cross-attention pathway.
"""

import os
import logging
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
log = logging.getLogger("rl")


class Config:
    train_path: str = "train.csv"
    test_path: str = "test.csv"
    mkt_path: str = "mkt_feats.csv"
    save_dir: str = "checkpoints"
    log_file: str = "training.log"
    lookback: int = 10
    episode_length: int = 240
    transaction_cost: float = 0.001
    n_actions: int = 3
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    mkt_dropout: float = 0.3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 50
    ppo_epochs: int = 4
    batch_size: int = 256
    inference_batch: int = 4096
    lr: float = 3e-4
    patience: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__class__.__annotations__}


# feature grouping

def split_groups(cols):
    groups = defaultdict(list)
    for i, c in enumerate(cols):
        groups[c.split(".")[0] if "." in c else c.split("_")[0]].append(i)
    stk, mkt = {}, {}
    for name, idx in groups.items():
        (mkt if name.startswith("mkt") else stk)[name] = idx
    stk_idx = sorted(i for v in stk.values() for i in v)
    mkt_idx = sorted(i for v in mkt.values() for i in v)
    return stk, mkt, stk_idx, mkt_idx


# vectorized environment

class VecEnv:
    def __init__(self, episodes, lookback, tx_cost):
        self.N = len(episodes)
        self.lookback = lookback
        self.tx_cost = tx_cost
        self.features = np.stack([e["features"] for e in episodes])
        self.log_returns = np.stack([e["log_returns"] for e in episodes])
        self.T = self.features.shape[1]
        self.n_steps = self.T - lookback

    def reset(self):
        self.t = self.lookback
        self.pos = np.zeros(self.N, dtype=np.float64)
        return self.features[:, self.t - self.lookback: self.t]

    def step(self, actions):
        new_pos = actions.astype(np.float64) - 1.0
        rewards = (self.pos * self.log_returns[:, self.t]
                   - self.tx_cost * np.abs(new_pos - self.pos)).astype(np.float32)
        self.pos = new_pos
        self.t += 1
        done = self.t >= self.T
        return (None if done else
                self.features[:, self.t - self.lookback: self.t]), rewards, done


# model

class GroupProjection(nn.Module):
    def __init__(self, group_indices, d_model):
        super().__init__()
        self.names = sorted(group_indices)
        n = len(self.names)
        base, rem = d_model // n, d_model % n
        self.proj = nn.ModuleDict()
        for i, name in enumerate(self.names):
            self.proj[name] = nn.Linear(len(group_indices[name]),
                                        base + (1 if i < rem else 0))
            self.register_buffer(f"i_{name}",
                                 torch.tensor(group_indices[name], dtype=torch.long))

    def forward(self, x):
        return torch.cat([self.proj[n](x[..., getattr(self, f"i_{n}")])
                          for n in self.names], dim=-1)


class Policy(nn.Module):
    def __init__(self, stk_groups, n_mkt, stk_idx, mkt_idx, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("stk_idx", torch.tensor(stk_idx, dtype=torch.long))
        self.register_buffer("mkt_idx", torch.tensor(mkt_idx, dtype=torch.long))
        self.has_mkt = n_mkt > 0

        self.stk_proj = GroupProjection(stk_groups, cfg.d_model)
        self.stk_norm = nn.LayerNorm(cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.lookback, cfg.d_model) * 0.02)

        if self.has_mkt:
            self.mkt_proj = nn.Sequential(
                nn.Linear(n_mkt, cfg.d_model), nn.LayerNorm(cfg.d_model))
            self.cross_attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(cfg.d_model)

        enc = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff,
            cfg.dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, cfg.n_layers)

        self.pi = nn.Sequential(
            nn.Linear(cfg.d_model, 64), nn.ReLU(), nn.Linear(64, cfg.n_actions))
        self.vf = nn.Sequential(
            nn.Linear(cfg.d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.stk_norm(self.stk_proj(x[..., self.stk_idx])) + self.pos_emb
        if self.has_mkt:
            m = self.mkt_proj(x[..., self.mkt_idx])
            if self.training and self.cfg.mkt_dropout > 0:
                mask = torch.bernoulli(torch.full(
                    (m.shape[0], 1, 1), 1 - self.cfg.mkt_dropout,
                    device=m.device, dtype=m.dtype))
                m = m * mask / (1 - self.cfg.mkt_dropout)
            out, _ = self.cross_attn(query=h, key=m, value=m)
            h = self.cross_norm(h + out)
        h = self.transformer(h).mean(dim=1)
        return self.pi(h), self.vf(h).squeeze(-1)


def get_action(model, states, deterministic=False):
    logits, values = model(states)
    dist = torch.distributions.Categorical(logits=logits)
    actions = logits.argmax(-1) if deterministic else dist.sample()
    return actions, dist.log_prob(actions), values


def eval_actions(model, states, actions):
    logits, values = model(states)
    dist = torch.distributions.Categorical(logits=logits)
    return dist.log_prob(actions), dist.entropy(), values


# rollout buffer

class Buffer:
    def __init__(self):
        self.s, self.a, self.r, self.lp, self.v, self.d = [], [], [], [], [], []

    def add_episode(self, states, actions, rewards, log_probs, values):
        n = len(rewards)
        self.s.extend(states); self.a.extend(actions); self.r.extend(rewards)
        self.lp.extend(log_probs); self.v.extend(values)
        self.d.extend([False] * (n - 1) + [True])

    def tensors(self, device, gamma, lam):
        v = np.array(self.v, dtype=np.float32)
        r = np.array(self.r, dtype=np.float32)
        d = np.array(self.d, dtype=np.float32)
        T = len(r)
        adv = np.zeros(T, dtype=np.float32)
        g = 0.0
        for t in reversed(range(T)):
            nv = 0.0 if t == T - 1 else v[t + 1] * (1 - d[t])
            delta = r[t] + gamma * nv - v[t]
            g = delta + gamma * lam * (1 - d[t]) * g
            adv[t] = g
        ret = adv + v
        std = adv.std()
        if std > 1e-8:
            adv = (adv - adv.mean()) / std
        return {
            "s": torch.tensor(np.array(self.s), dtype=torch.float32, device=device),
            "a": torch.tensor(np.array(self.a), dtype=torch.long, device=device),
            "lp": torch.tensor(np.array(self.lp), dtype=torch.float32, device=device),
            "adv": torch.tensor(adv, dtype=torch.float32, device=device),
            "ret": torch.tensor(ret, dtype=torch.float32, device=device),
        }

    def clear(self):
        self.__init__()


# data loading

def load_data(path):
    df = pd.read_csv(path).sort_values(["symbol", "date"]).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ("symbol", "date", "price")]
    data = {}
    for sym, g in df.groupby("symbol"):
        p = g["price"].values.astype(np.float64)
        lr = np.zeros(len(p), dtype=np.float32)
        lr[1:] = np.log(p[1:] / p[:-1]).astype(np.float32)
        data[str(sym)] = {
            "features": g[feat_cols].values.astype(np.float32),
            "log_returns": lr,
            "dates": g["date"].tolist(),
            "prices": p,
        }
    return data, feat_cols


def load_mkt_data(path):
    df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    cols = [c for c in df.columns if c != "date"]
    return {
        "features": df[cols].values.astype(np.float32),
        "date_to_idx": {d: i for i, d in enumerate(df["date"])},
        "cols": cols,
    }


# chunk generation

def _chunk_starts(args):
    sym, T, lookback, ep_len, seed = args
    cs = lookback + ep_len
    if T < cs:
        return sym, []
    rng = np.random.default_rng(seed)
    anchor = rng.integers(0, T)
    starts = set()
    s = anchor
    while s + cs <= T:
        starts.add(s); s += ep_len
    s = anchor - ep_len
    while s >= 0:
        if s + cs <= T:
            starts.add(s)
        s -= ep_len
    return sym, sorted(starts)


def gen_chunks(data, mkt, cfg, rng):
    syms = list(data)
    seeds = rng.integers(0, 2**63, size=len(syms))
    args = [(s, len(data[s]["features"]), cfg.lookback,
             cfg.episode_length, int(seeds[i])) for i, s in enumerate(syms)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_chunk_starts, args))

    cs = cfg.lookback + cfg.episode_length
    date_to_idx = mkt["date_to_idx"]
    mkt_feats = mkt["features"]
    eps = []
    for sym, starts in results:
        d = data[sym]
        for s in starts:
            chunk_dates = d["dates"][s:s+cs]
            mkt_rows = [date_to_idx.get(dt) for dt in chunk_dates]
            if any(r is None for r in mkt_rows):
                continue
            combined = np.concatenate(
                [d["features"][s:s+cs], mkt_feats[mkt_rows]], axis=1)
            eps.append({
                "symbol": sym, "features": combined,
                "log_returns": d["log_returns"][s:s+cs],
                "dates": chunk_dates, "prices": d["prices"][s:s+cs],
            })
    return eps


# rollout collection

def collect(model, episodes, cfg):
    buf = Buffer()
    model.eval()
    with torch.no_grad():
        for bs in range(0, len(episodes), cfg.inference_batch):
            batch = episodes[bs:bs+cfg.inference_batch]
            N = len(batch)
            env = VecEnv(batch, cfg.lookback, cfg.transaction_cost)
            states = env.reset()
            ns, nF = env.n_steps, states.shape[-1]
            aS = np.empty((ns, N, cfg.lookback, nF), dtype=np.float32)
            aA = np.empty((ns, N), dtype=np.int64)
            aR = np.empty((ns, N), dtype=np.float32)
            aL = np.empty((ns, N), dtype=np.float32)
            aV = np.empty((ns, N), dtype=np.float32)
            for t in range(ns):
                aS[t] = states
                act, lp, val = get_action(model, torch.from_numpy(states).to(cfg.device))
                an = act.cpu().numpy()
                aA[t], aL[t], aV[t] = an, lp.cpu().numpy(), val.cpu().numpy()
                states, aR[t], _ = env.step(an)
            for i in range(N):
                buf.add_episode(aS[:,i], aA[:,i], aR[:,i], aL[:,i], aV[:,i])
    return buf


# PPO update

def ppo_update(model, optimizer, buf, cfg):
    model.train()
    d = buf.tensors(cfg.device, cfg.gamma, cfg.gae_lambda)
    n = len(d["s"])
    tpl = tvl = te = 0.0
    nu = 0
    base = model.module if hasattr(model, "module") else model
    for _ in range(cfg.ppo_epochs):
        idx = torch.randperm(n, device=cfg.device)
        for start in range(0, n, cfg.batch_size):
            i = idx[start:min(start+cfg.batch_size, n)]
            lp, ent, val = eval_actions(model, d["s"][i], d["a"][i])
            ratio = torch.exp(lp - d["lp"][i])
            adv = d["adv"][i]
            pl = -torch.min(ratio * adv,
                            torch.clamp(ratio, 1-cfg.clip_eps, 1+cfg.clip_eps) * adv).mean()
            vl = F.mse_loss(val, d["ret"][i])
            loss = pl + cfg.value_coeff * vl - cfg.entropy_coeff * ent.mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), cfg.max_grad_norm)
            optimizer.step()
            tpl += pl.item(); tvl += vl.item(); te += ent.mean().item(); nu += 1
    return {"pl": tpl/max(nu,1), "vl": tvl/max(nu,1), "ent": te/max(nu,1)}


# buy and hold worker

def _bh_batch(args):
    lr_list, lookback, tx_cost = args
    lr = np.stack(lr_list)
    N, T = lr.shape
    ns = T - lookback
    pos = np.zeros(N, dtype=np.float64)
    rw = np.empty((ns, N), dtype=np.float32)
    for t in range(ns):
        new = np.ones(N, dtype=np.float64)
        rw[t] = (pos * lr[:, lookback+t] - tx_cost * np.abs(new - pos)).astype(np.float32)
        pos = new
    return rw


# evaluation

def evaluate(model, episodes, cfg, verbose=False):
    ne = len(episodes)
    if ne == 0:
        return {"model_return": 0, "bh_return": 0, "excess": 0,
                "beat_rate": 0, "metrics": [], "rows": []}
    ns = cfg.episode_length
    mr = np.empty((ns, ne), dtype=np.float32)
    mp = np.empty((ns, ne), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for bs in range(0, ne, cfg.inference_batch):
            batch = episodes[bs:bs+cfg.inference_batch]
            N = len(batch)
            env = VecEnv(batch, cfg.lookback, cfg.transaction_cost)
            st = env.reset()
            for t in range(ns):
                act, _, _ = get_action(model, torch.from_numpy(st).to(cfg.device), True)
                an = act.cpu().numpy()
                mp[t, bs:bs+N] = an - 1
                st, rw, _ = env.step(an)
                mr[t, bs:bs+N] = rw

    bsz = max(1, ne // N_WORKERS)
    bh_args = [([e["log_returns"] for e in episodes[i:i+bsz]],
                cfg.lookback, cfg.transaction_cost) for i in range(0, ne, bsz)]
    br = np.empty((ns, ne), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, rw in enumerate(pool.map(_bh_batch, bh_args)):
            s = i * bsz
            br[:, s:s+rw.shape[1]] = rw

    rows, metrics = [], []
    for i in range(ne):
        m, b = mr[:, i], br[:, i]
        ms, bs_ = m.std(), b.std()
        mret = float(m.mean() / ms) if ms > 1e-8 else 0.0
        bret = float(b.mean() / bs_) if bs_ > 1e-8 else 0.0
        pos = mp[:, i]
        nt = int(np.sum(np.diff(pos) != 0))
        td = (m != 0).sum()
        wr = float((m[m != 0] > 0).mean()) if td > 0 else 0.0
        cum = np.cumsum(m)
        dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
        metrics.append({"symbol": episodes[i]["symbol"], "model": mret, "bh": bret,
                        "excess": mret - bret, "dd": dd, "trades": nt, "wr": wr})
        dates = episodes[i]["dates"][cfg.lookback:]
        prices = episodes[i]["prices"][cfg.lookback:]
        for t in range(ns):
            rows.append({"symbol": episodes[i]["symbol"], "date": dates[t],
                         "price": prices[t], "position": int(pos[t]),
                         "model_pnl": float(m[t]), "bh_pnl": float(b[t])})

    mrets = [x["model"] for x in metrics]
    brets = [x["bh"] for x in metrics]
    excs = [x["excess"] for x in metrics]
    beat = float(np.mean([e > 0 for e in excs]))

    if verbose:
        log.info("")
        log.info(f"{'Sym':>8} {'Model':>8} {'B&H':>8} {'Excess':>8} "
                 f"{'DD':>7} {'Trd':>5} {'WR':>6}")
        for x in metrics:
            log.info(f"{x['symbol']:>8} {x['model']:8.4f} {x['bh']:8.4f} "
                     f"{x['excess']:+8.4f} {x['dd']:7.4f} {x['trades']:5d} {x['wr']:6.3f}")
        log.info("")
        log.info(f"{'':>10} {'Mean':>8} {'Median':>8}")
        log.info(f"{'Model':>10} {np.mean(mrets):8.4f} {np.median(mrets):8.4f}")
        log.info(f"{'B&H':>10} {np.mean(brets):8.4f} {np.median(brets):8.4f}")
        log.info(f"{'Excess':>10} {np.mean(excs):+8.4f} {np.median(excs):+8.4f}")
        log.info(f"  Beat rate: {beat:.1%}  |  Episodes: {ne}")

    return {"model_return": float(np.mean(mrets)), "bh_return": float(np.mean(brets)),
            "excess": float(np.mean(excs)), "beat_rate": beat,
            "metrics": metrics, "rows": rows}


# plotting

def plot(h, path):
    ep = range(1, len(h["tr_mod"]) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0,0].plot(ep, h["tr_mod"], "b-", label="Train", alpha=.8)
    ax[0,0].plot(ep, h["vl_mod"], "r-", label="Val", alpha=.8)
    ax[0,0].plot(ep, h["tr_bh"], "b--", label="Train B&H", alpha=.5)
    ax[0,0].plot(ep, h["vl_bh"], "r--", label="Val B&H", alpha=.5)
    ax[0,0].set_ylabel("Return (mean/std)"); ax[0,0].set_title("Return")
    ax[0,0].legend(fontsize=8); ax[0,0].grid(True, alpha=.3)

    ax[0,1].plot(ep, h["tr_exc"], "b-", label="Train", alpha=.8)
    ax[0,1].plot(ep, h["vl_exc"], "r-", label="Val", alpha=.8)
    ax[0,1].axhline(0, color="k", ls="--", alpha=.3)
    ax[0,1].set_ylabel("Excess over B&H"); ax[0,1].set_title("Excess")
    ax[0,1].legend(fontsize=8); ax[0,1].grid(True, alpha=.3)

    ax[1,0].plot(ep, h["pl"], "b-", label="Policy", alpha=.8)
    ax[1,0].plot(ep, h["vl_loss"], "r-", label="Value", alpha=.8)
    ax[1,0].set_xlabel("Epoch"); ax[1,0].set_ylabel("Loss")
    ax[1,0].set_title("Losses"); ax[1,0].legend(fontsize=8); ax[1,0].grid(True, alpha=.3)

    ax[1,1].plot(ep, h["ent"], "g-", label="Entropy", alpha=.8)
    a2 = ax[1,1].twinx()
    a2.plot(ep, h["beat"], "r-", label="Beat rate", alpha=.8)
    a2.set_ylabel("Beat rate", color="r"); a2.set_ylim(0, 1)
    ax[1,1].set_xlabel("Epoch"); ax[1,1].set_ylabel("Entropy", color="g")
    ax[1,1].set_title("Entropy & Beat Rate"); ax[1,1].grid(True, alpha=.3)
    l1, lb1 = ax[1,1].get_legend_handles_labels()
    l2, lb2 = a2.get_legend_handles_labels()
    ax[1,1].legend(l1+l2, lb1+lb2, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# training loop

def train(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg.save_dir, cfg.log_file),
        filemode="w", level=logging.INFO,
        format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    train_data, stk_cols = load_data(cfg.train_path)
    test_data, _ = load_data(cfg.test_path)
    mkt = load_mkt_data(cfg.mkt_path)
    all_cols = stk_cols + mkt["cols"]
    log.info(f"Train: {len(train_data)} symbols, {len(stk_cols)} stock feats, "
             f"{len(mkt['cols'])} market feats")
    log.info(f"Test:  {len(test_data)} symbols")

    val_eps = gen_chunks(test_data, mkt, cfg,
                         np.random.default_rng(cfg.seed + 1000))
    log.info(f"Val episodes: {len(val_eps)} (fixed)")

    stk_g, mkt_g, stk_i, mkt_i = split_groups(all_cols)
    log.info(f"Stock groups: {len(stk_g)} ({len(stk_i)} feats)  "
             f"Market groups: {len(mkt_g)} ({len(mkt_i)} feats)  "
             f"Mkt dropout: {cfg.mkt_dropout}")

    model = Policy(stk_g, len(mkt_i), stk_i, mkt_i, cfg).to(cfg.device)
    n_gpus = torch.cuda.device_count()
    dp = nn.DataParallel(model) if n_gpus > 1 else model
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}  "
             f"Device: {f'{n_gpus} GPUs' if n_gpus > 1 else cfg.device}  "
             f"Workers: {N_WORKERS}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    hist = defaultdict(list)
    best_val, patience_ctr = -float("inf"), 0

    hdr = (f"{'Ep':>4} {'PL':>7} {'VL':>7} {'Ent':>6} "
           f"{'TrMod':>7} {'TrExc':>7} {'VlMod':>7} {'VlExc':>7} {'Beat':>6} {'#Ep':>5}")
    log.info(f"Training up to {cfg.n_epochs} epochs, patience {cfg.patience}")
    log.info(hdr)

    for epoch in range(1, cfg.n_epochs + 1):
        for pg in opt.param_groups:
            pg["lr"] = cfg.lr * (1 - (epoch - 1) / cfg.n_epochs)

        tr_eps = gen_chunks(train_data, mkt, cfg, rng)
        buf = collect(dp, tr_eps, cfg)
        loss = ppo_update(dp, opt, buf, cfg)
        buf.clear()

        tr = evaluate(dp, tr_eps, cfg)
        vl = evaluate(dp, val_eps, cfg)

        for k, v in [("tr_mod", tr["model_return"]), ("tr_bh", tr["bh_return"]),
                      ("tr_exc", tr["excess"]), ("vl_mod", vl["model_return"]),
                      ("vl_bh", vl["bh_return"]), ("vl_exc", vl["excess"]),
                      ("beat", vl["beat_rate"]), ("pl", loss["pl"]),
                      ("vl_loss", loss["vl"]), ("ent", loss["ent"])]:
            hist[k].append(v)

        log.info(f"{epoch:4d} {loss['pl']:7.4f} {loss['vl']:7.4f} {loss['ent']:6.4f} "
                 f"{tr['model_return']:7.4f} {tr['excess']:+7.4f} "
                 f"{vl['model_return']:7.4f} {vl['excess']:+7.4f} "
                 f"{vl['beat_rate']:6.1%} {len(tr_eps):5d}")

        _save(model, cfg, all_cols, stk_g, mkt_g, stk_i, mkt_i,
              os.path.join(cfg.save_dir, "model_latest.pt"), epoch)
        plot(dict(hist), os.path.join(cfg.save_dir, "training_curves.png"))

        if vl["model_return"] > best_val:
            best_val = vl["model_return"]
            patience_ctr = 0
            _save(model, cfg, all_cols, stk_g, mkt_g, stk_i, mkt_i,
                  os.path.join(cfg.save_dir, "model_best.pt"), epoch)
            log.info(f"  new best val {best_val:.4f}")
        else:
            patience_ctr += 1

        if patience_ctr >= cfg.patience:
            log.info(f"Early stop at epoch {epoch} (patience {cfg.patience})")
            break

    log.info(f"Done. Best val: {best_val:.4f}")

    ckpt = torch.load(os.path.join(cfg.save_dir, "model_best.pt"),
                      map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    log.info("FINAL EVAL (best model, fresh chunks)")
    final_eps = gen_chunks(test_data, mkt, cfg,
                           np.random.default_rng(cfg.seed + 9999))
    log.info(f"Episodes: {len(final_eps)}")
    res = evaluate(dp, final_eps, cfg, verbose=True)
    pd.DataFrame(res["rows"]).to_csv(
        os.path.join(cfg.save_dir, "test_results.csv"), index=False)
    log.info(f"Results saved to {os.path.join(cfg.save_dir, 'test_results.csv')}")


def _save(model, cfg, all_cols, stk_g, mkt_g, stk_i, mkt_i, path, epoch):
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                "feature_cols": all_cols, "stock_groups": stk_g,
                "market_groups": mkt_g, "stock_indices": stk_i,
                "market_indices": mkt_i, "epoch": epoch}, path)


# CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train"); p.add_argument("--test"); p.add_argument("--mkt")
    p.add_argument("--save_dir"); p.add_argument("--log_file")
    p.add_argument("--epochs", type=int); p.add_argument("--lr", type=float)
    p.add_argument("--lookback", type=int); p.add_argument("--episode_length", type=int)
    p.add_argument("--tx_cost", type=float); p.add_argument("--mkt_dropout", type=float)
    p.add_argument("--patience", type=int); p.add_argument("--seed", type=int)
    a = p.parse_args()
    cfg = Config()
    if a.train: cfg.train_path = a.train
    if a.test: cfg.test_path = a.test
    if a.mkt: cfg.mkt_path = a.mkt
    if a.save_dir: cfg.save_dir = a.save_dir
    if a.log_file: cfg.log_file = a.log_file
    if a.epochs: cfg.n_epochs = a.epochs
    if a.lr: cfg.lr = a.lr
    if a.lookback: cfg.lookback = a.lookback
    if a.episode_length: cfg.episode_length = a.episode_length
    if a.tx_cost: cfg.transaction_cost = a.tx_cost
    if a.mkt_dropout is not None: cfg.mkt_dropout = a.mkt_dropout
    if a.patience: cfg.patience = a.patience
    if a.seed: cfg.seed = a.seed
    train(cfg)


if __name__ == "__main__":
    main()
