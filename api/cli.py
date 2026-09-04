import argparse
import os
import sys


def _ensure_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def cmd_serve(args):
    _ensure_path()
    import uvicorn

    from api.main import app
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_version(args):
    _ensure_path()
    from api.version import __version__
    print(f"lexrag {__version__}")


def cmd_ingest(args):
    _ensure_path()
    from scripts.ingest import ingest_pdf, ingest_text
    path = args.path
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return 1
    name = os.path.basename(path)
    lower = name.lower()
    jurisdiction = "India" if "india" in lower else ("UAE" if "uae" in lower else "Both")
    meta = {
        "source": name,
        "source_type": "statute",
        "jurisdiction": jurisdiction,
        "doc_title": os.path.splitext(name)[0].replace("_", " ").title(),
        "date": "imported",
        "url": "",
    }
    if lower.endswith(".pdf"):
        ingest_pdf(path, meta)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            ingest_text(f.read(), meta)
    print(f"Done: {path}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="lexrag", description="LexRAG — Legal Intelligence Terminal")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run the API + UI server")
    s.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.set_defaults(fn=cmd_serve)

    v = sub.add_parser("version", help="Print version")
    v.set_defaults(fn=cmd_version)

    i = sub.add_parser("ingest", help="Ingest a text/PDF file into the local index")
    i.add_argument("path")
    i.set_defaults(fn=cmd_ingest)

    args = p.parse_args(argv)
    rc = args.fn(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
