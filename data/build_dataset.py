#!/usr/bin/env python3
"""Rebuild the Appendix B datasets from Cremona's ecdata.

Produces one row per isogeny class of conductor < 500000 (2,100,011 classes,
the set called E_2 in the paper):

  data/aps100_ainvs.parquet       input = (a_p)_{p<100}       target = w1,w2
  data/aps100_ainvs_full.parquet  input = (a_p)_{p<100}, N    target = w1,w2,w3

Source: https://github.com/JohnCremona/ecdata (Artistic License 2.0)

  allcurves/allcurves.<lo>-<hi>   conductor c_class c_number [a1,a2,a3,a4,a6] rank torsion
  alllabels/alllabels.<lo>-<hi>   conductor c_class c_number conductor lmfdb_class lmfdb_number

LMFDB numbers its curves differently from Cremona (e.g. Cremona 100a3 is
LMFDB 100.a1), so alllabels is needed to pick curve number 1 of each LMFDB
isogeny class -- the representative used to define (w1,w2,w3).

a_p is not stored in LMFDB and is recomputed here from the global minimal
model; see src/data/frobenius.py.

Usage
-----
    python data/build_dataset.py                      # downloads ~190 MB, caches it
    python data/build_dataset.py --ecdata /path/to/ecdata   # use a local clone
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.frobenius import ap_matrix, PRIMES_UNDER_100  # noqa: E402

RAW = "https://raw.githubusercontent.com/JohnCremona/ecdata/master"
BLOCK = 10000


def blocks(max_conductor):
    for lo in range(0, max_conductor, BLOCK):
        yield f"{lo:05d}-{lo + BLOCK - 1:05d}"


def fetch(kind, block, cache_dir, ecdata=None):
    """Return the local path to one ecdata file, downloading it if needed."""
    name = f"{kind}.{block}"
    if ecdata:
        local = os.path.join(ecdata, kind, name)
        if not os.path.exists(local):
            raise FileNotFoundError(local)
        return local
    os.makedirs(os.path.join(cache_dir, kind), exist_ok=True)
    local = os.path.join(cache_dir, kind, name)
    if not os.path.exists(local):
        url = f"{RAW}/{kind}/{name}"
        print(f"  downloading {kind}/{name}", flush=True)
        urllib.request.urlretrieve(url, local + ".part")
        os.replace(local + ".part", local)
    return local


def read_block(block, cache_dir, ecdata):
    """One block of conductors -> DataFrame[lmfdb_iso, conductor, a1..a6], LMFDB curve 1 only."""
    labels = {}
    with open(fetch("alllabels", block, cache_dir, ecdata)) as fh:
        for line in fh:
            N, ccls, cnum, _, lcls, lnum = line.split()
            if lnum == "1":                      # representative of the isogeny class
                labels[(N, ccls, cnum)] = lcls

    rows = []
    with open(fetch("allcurves", block, cache_dir, ecdata)) as fh:
        for line in fh:
            N, ccls, cnum, ainvs = line.split()[:4]
            lcls = labels.get((N, ccls, cnum))
            if lcls is None:
                continue
            rows.append((f"{N}.{lcls}", int(N), *[int(v) for v in ainvs[1:-1].split(",")]))

    return pd.DataFrame(rows, columns=["lmfdb_iso", "conductor",
                                       "a1", "a2", "a3", "a4", "a6"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ecdata", default=None,
                    help="path to a local ecdata clone (default: download the two "
                         "directories we need into --cache-dir)")
    ap.add_argument("--cache-dir", default="data/ecdata")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max-conductor", type=int, default=500000)
    args = ap.parse_args()

    parts = []
    for block in blocks(args.max_conductor):
        print(f"conductor {block}", flush=True)
        parts.append(read_block(block, args.cache_dir, args.ecdata))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.conductor < args.max_conductor].reset_index(drop=True)
    print(f"\n{len(df):,} isogeny classes, conductor {df.conductor.min()}"
          f"-{df.conductor.max()}")

    print(f"computing a_p for p < 100 ({len(PRIMES_UNDER_100)} primes) ...", flush=True)
    aps = ap_matrix(df[["a1", "a2", "a3", "a4", "a6"]].to_numpy())

    def join(mat):
        return [", ".join(map(str, row)) for row in mat.tolist()]

    ap_str = join(aps)
    w = df[["a1", "a2", "a3"]].to_numpy()
    os.makedirs(args.out_dir, exist_ok=True)

    # (a_p)_{p<100} -> (w1, w2)
    p1 = os.path.join(args.out_dir, "aps100_ainvs.parquet")
    pd.DataFrame({"lmfdb_iso": df.lmfdb_iso, "input": ap_str,
                  "target": join(w[:, :2])}).to_parquet(p1, index=False)
    print(f"wrote {p1}")

    # ((a_p)_{p<100}, N) -> (w1, w2, w3)
    p2 = os.path.join(args.out_dir, "aps100_ainvs_full.parquet")
    pd.DataFrame({"lmfdb_iso": df.lmfdb_iso,
                  "input": [f"{s}, {n}" for s, n in zip(ap_str, df.conductor)],
                  "target": join(w), "conductor": df.conductor}).to_parquet(p2, index=False)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
