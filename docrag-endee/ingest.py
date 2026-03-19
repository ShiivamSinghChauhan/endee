#!/usr/bin/env python3
"""
CLI Ingestion Tool
------------------
Usage:
    python ingest.py data/sample_docs/intro_to_rag.txt
    python ingest.py data/docs/
    python ingest.py data/paper.pdf --index my-index
"""

import argparse, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Endee.")
    parser.add_argument("path",    help="File or directory to ingest")
    parser.add_argument("--index", default=None, help="Endee index name")
    args = parser.parse_args()

    from rag.ingestion import Ingestor
    kwargs = {}
    if args.index:
        kwargs["index_name"] = args.index

    ingestor = Ingestor(**kwargs)
    target   = Path(args.path)

    if target.is_dir():
        total = ingestor.ingest_directory(target)
    elif target.is_file():
        total = ingestor.ingest_file(target)
    else:
        print(f"❌ Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Done. {total} chunks stored in Endee.")

if __name__ == "__main__":
    main()
