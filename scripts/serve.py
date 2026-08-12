"""薄 CLI：起 QuantAI FastAPI 服务（quantai.api）。

示例：
    python scripts/serve.py --host 0.0.0.0 --port 8000

只暴露真实 endpoint（health / live / feedback / evolution / chat）。chat 需注入 LLM，
默认无 -> 返回 503（不伪造）。文档见 http://localhost:8000/docs。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description="QuantAI API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="开发热重载")
    args = p.parse_args()

    import uvicorn

    uvicorn.run("quantai.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
