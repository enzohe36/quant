"""
Robust replacement for akshare's stock_zh_a_spot_em().
Uses Tencent's qt.gtimg.cn API instead of Eastmoney's push2 endpoint,
which is geo-blocked outside China.

No cookie or authentication required.
"""

import random
import time
from typing import List, Tuple

import pandas as pd
import requests

_BATCH_SIZE = 500
_SLEEP_RANGE = (0.03, 0.08)
_TIMEOUT = 15
_RETRIES = 2

# Tencent field indices (0-indexed, tilde-separated)
_F_NAME = 1
_F_CODE = 2
_F_CLOSE = 3       # 最新价
_F_PREV_CLOSE = 4  # 昨收
_F_OPEN = 5        # 今开
_F_VOLUME = 6      # 成交量 (手)
_F_CHANGE_AMT = 31 # 涨跌额
_F_CHANGE_PCT = 32 # 涨跌幅 (%)
_F_HIGH = 33       # 最高
_F_LOW = 34        # 最低
_F_AMOUNT = 37     # 成交额 (万元)
_F_TURNOVER = 38   # 换手率 (%)
_F_PE = 39         # 市盈率-动态
_F_AMPLITUDE = 43  # 振幅 (%)
_F_MKT_CAP = 44    # 总市值 (亿元)
_F_FLOAT_CAP = 45  # 流通市值 (亿元)
_F_PB = 46         # 市净率
_F_VOL_RATIO = 49  # 量比

# akshare-compatible output columns
KEEP_COLS = [
    "序号", "代码", "名称", "最新价", "涨跌幅", "涨跌额",
    "成交量", "成交额", "振幅", "最高", "最低", "今开", "昨收",
    "量比", "换手率", "市盈率-动态", "市净率", "总市值", "流通市值",
    "涨速", "5分钟涨跌", "60日涨跌幅", "年初至今涨跌幅",
]


def _generate_codes() -> List[str]:
    """All possible A-share code prefixes for Tencent query."""
    codes = []
    # SH main board: 600xxx-605xxx
    for prefix in range(600, 606):
        codes += [f"sh{prefix}{i:03d}" for i in range(1000)]
    # SH STAR: 688xxx-689xxx
    for prefix in range(688, 690):
        codes += [f"sh{prefix}{i:03d}" for i in range(1000)]
    # SZ main board: 000xxx-003xxx
    for prefix in range(0, 4):
        codes += [f"sz{prefix:03d}{i:03d}" for i in range(1000)]
    # SZ ChiNext: 300xxx-301xxx
    for prefix in range(300, 302):
        codes += [f"sz{prefix}{i:03d}" for i in range(1000)]
    return codes


def _parse_tencent_line(content: str) -> dict:
    """Parse one Tencent quote line into a dict with akshare-compatible keys."""
    f = content.split("~")
    if len(f) < 50:
        return {}

    def _num(idx):
        try:
            return float(f[idx])
        except (IndexError, ValueError):
            return float("nan")

    return {
        "代码": f[_F_CODE],
        "名称": f[_F_NAME],
        "最新价": _num(_F_CLOSE),
        "涨跌幅": _num(_F_CHANGE_PCT),
        "涨跌额": _num(_F_CHANGE_AMT),
        "成交量": _num(_F_VOLUME),            # 手
        "成交额": _num(_F_AMOUNT) * 10000,     # 万元 → 元
        "振幅": _num(_F_AMPLITUDE),
        "最高": _num(_F_HIGH),
        "最低": _num(_F_LOW),
        "今开": _num(_F_OPEN),
        "昨收": _num(_F_PREV_CLOSE),
        "量比": _num(_F_VOL_RATIO),
        "换手率": _num(_F_TURNOVER),
        "市盈率-动态": _num(_F_PE),
        "市净率": _num(_F_PB),
        "总市值": _num(_F_MKT_CAP) * 1e8,     # 亿元 → 元
        "流通市值": _num(_F_FLOAT_CAP) * 1e8,  # 亿元 → 元
        "涨速": float("nan"),
        "5分钟涨跌": float("nan"),
        "60日涨跌幅": float("nan"),
        "年初至今涨跌幅": float("nan"),
    }


def stock_zh_a_spot_em_robust() -> pd.DataFrame:
    """
    Fetch realtime A-share quotes from Tencent.
    Drop-in replacement for akshare's stock_zh_a_spot_em().
    """
    codes = _generate_codes()
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    rows = []
    lo, hi = _SLEEP_RANGE

    for i in range(0, len(codes), _BATCH_SIZE):
        batch = codes[i : i + _BATCH_SIZE]
        query = ",".join(batch)

        last_err = None
        for _ in range(_RETRIES + 1):
            try:
                resp = sess.get(
                    f"http://qt.gtimg.cn/q={query}", timeout=_TIMEOUT
                )
                break
            except Exception as e:
                last_err = e
                time.sleep(random.uniform(0.3, 0.7))
        else:
            raise last_err

        raw = resp.content.decode("gbk", errors="replace")
        for line in raw.split(";"):
            line = line.strip()
            if not line or '"' not in line:
                continue
            content = line.split('"')[1]
            if not content or content.count("~") < 30:
                continue
            row = _parse_tencent_line(content)
            if row and row["代码"]:
                rows.append(row)

        if hi > 0:
            time.sleep(random.uniform(lo, hi))

    if not rows:
        return pd.DataFrame(columns=KEEP_COLS)

    df = pd.DataFrame(rows)
    df.sort_values(by="涨跌幅", ascending=False, inplace=True, ignore_index=True)
    df.insert(0, "序号", range(1, len(df) + 1))
    df = df[KEEP_COLS]
    return df
