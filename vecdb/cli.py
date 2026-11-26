"""Command line interface: ``python -m vecdb serve`` / ``python -m vecdb info``."""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vecdb", description="vector database")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--data", default="./vecdb-data", help="directory holding collections")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    i = sub.add_parser("info", help="open a collection directory (runs recovery) and print stats")
    i.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(args.data), host=args.host, port=args.port)
    elif args.cmd == "info":
        from .collection import Collection

        col = Collection(args.path)
        print(json.dumps({"recovery": col.recovery, **col.stats()}, indent=2))
        col.close()


if __name__ == "__main__":
    main()
