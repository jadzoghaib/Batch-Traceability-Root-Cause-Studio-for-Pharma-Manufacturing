"""
Fetch the source dataset from figshare.

Žagar, J. & Mihelič, J. "Big data collection in pharmaceutical manufacturing and
its use for product quality predictions." Scientific Data 9, 99 (2022).
Collection DOI 10.6084/m9.figshare.c.5645578 — CC-BY 4.0.

    python scripts/download_data.py            # everything (~30 MB download)
    python scripts/download_data.py --no-ts    # skip the time-series archive
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from batchlens import config as C

BASE = "https://ndownloader.figshare.com/files/"


def fetch(name: str, file_id: str, dest: Path) -> None:
    out = dest / name
    if out.exists() and out.stat().st_size > 0:
        print(f"  ✓ {name} already present ({out.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  ↓ {name} …", end="", flush=True)

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, 100 * blocks * bs / total)
            print(f"\r  ↓ {name} … {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(BASE + file_id, out, reporthook=hook)
    print(f"\r  ✓ {name} ({out.stat().st_size / 1e6:.1f} MB)        ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ts", action="store_true",
                    help="skip Process.zip (the 10-second time series)")
    args = ap.parse_args()

    C.RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {C.RAW}")
    for name, fid in C.FIGSHARE_FILES.items():
        if args.no_ts and name == "Process.zip":
            print(f"  – {name} skipped")
            continue
        fetch(name, fid, C.RAW)

    print("\nNext:  python -m batchlens.etl")


if __name__ == "__main__":
    main()
