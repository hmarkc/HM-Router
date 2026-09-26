"""Train and evaluate hm-router, Avengers and Random on hm-router-bench
(70/30 query split per seed). Prints oracle-normalized accuracy by default.

    python -m src.run                          # all 12 benchmarks
    python -m src.run --benchmarks gaia scicode
    python -m src.run --raw                    # unnormalized accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import avengers_select, random_accuracy
from src.data import ROOT, SEEDS, benchmarks, load_benchmark, split_queries
from src.hm_router import hm_router_select

COLUMNS = {"random": "Random", "avengers": "Avengers", "hm-router": "hm-router",
           "oracle": "Oracle"}


def evaluate(name: str, seeds: list[int]) -> list[dict]:
    bench = load_benchmark(name)
    acc, emb = bench.accuracy, bench.embeddings
    rows = []
    for seed in seeds:
        train, test = split_queries(len(bench.query_ids), seed)
        rows_test = np.arange(len(test))
        picks = {
            "avengers": avengers_select(emb, acc, train, test, seed),
            "hm-router": hm_router_select(emb, acc, bench.routes, train, test, seed),
            "oracle": acc[test].argmax(axis=1),
        }
        scores = {"random": random_accuracy(acc, test)}
        scores |= {p: float(acc[test][rows_test, s].mean()) for p, s in picks.items()}
        for policy, accuracy in scores.items():
            rows.append({"benchmark": name, "seed": seed, "policy": policy,
                         "accuracy": accuracy, "routes": len(bench.routes)})
        print(f"{name:22s} seed={seed:5d} "
              + " ".join(f"{p}={a:.4f}" for p, a in scores.items()), flush=True)
    return rows


def table(results: pd.DataFrame, raw: bool) -> pd.DataFrame:
    results = results.copy()
    if not raw:
        oracle = results.loc[results["policy"].eq("oracle")].set_index(
            ["benchmark", "seed"])["accuracy"]
        keys = pd.MultiIndex.from_frame(results[["benchmark", "seed"]])
        results["accuracy"] /= oracle.reindex(keys).to_numpy()
    frame = results.pivot_table(index="benchmark", columns="policy",
                                values="accuracy", aggfunc="mean", sort=False)
    frame = frame[list(COLUMNS)].rename(columns=COLUMNS)
    frame.loc["mean"] = frame.mean()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "results.csv")
    parser.add_argument("--raw", action="store_true",
                        help="report raw accuracy instead of oracle-normalized")
    args = parser.parse_args()

    rows = [row for name in args.benchmarks or benchmarks()
            for row in evaluate(name, args.seeds)]
    results = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    frame = table(results, args.raw)
    print("\n| benchmark | " + " | ".join(frame.columns) + " |")
    print("|---|" + "---:|" * len(frame.columns))
    for name, row in frame.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")


if __name__ == "__main__":
    main()
