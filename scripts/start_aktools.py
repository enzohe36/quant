"""
Start aktools with a patched stock_zh_a_spot_em that uses Tencent's API
instead of Eastmoney's geo-blocked push2 endpoint.

Usage:
    conda activate quant
    python scripts/start_aktools.py
"""

import sys

import akshare as ak

from em_realtime import stock_zh_a_spot_em_robust

# Monkey-patch akshare so aktools' eval("ak.stock_zh_a_spot_em()") uses our version.
ak.stock_zh_a_spot_em = stock_zh_a_spot_em_robust

import uvicorn
from aktools.main import app

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8080
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--host", "-H") and i < len(sys.argv) - 1:
            host = sys.argv[i + 1]
        elif arg in ("--port", "-P") and i < len(sys.argv) - 1:
            port = int(sys.argv[i + 1])

    print(f"aktools (patched) running at http://{host}:{port}/")
    print(f"akshare version: {ak.__version__}")
    uvicorn.run(app, host=host, port=port)
