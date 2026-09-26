"""Loads hm-router-bench as dense query x route accuracy matrices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
SEEDS = [42, 999, 2024, 2025, 3407]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


@dataclass
class Benchmark:
    name: str
    query_ids: np.ndarray    # (Q,)
    texts: list[str]         # (Q,)
    routes: pd.DataFrame     # (R,) columns: model, harness, scaffold
    accuracy: np.ndarray     # (Q, R) per-query per-route accuracy
    embeddings: np.ndarray   # (Q, 4096) frozen query embeddings


def read_csv(name: str) -> pd.DataFrame:
    # round_trip: exact float parsing
    return pd.read_csv(DATA / name, dtype={"query_id": str},
                       keep_default_na=False, float_precision="round_trip")


def benchmarks() -> list[str]:
    return list(read_csv("hm_router_bench.csv")["benchmark"].unique())


def load_benchmark(name: str) -> Benchmark:
    data = read_csv("hm_router_bench.csv")
    data = data.loc[data["benchmark"].eq(name)]
    queries = read_csv("queries.csv")
    queries = queries.loc[queries["benchmark"].eq(name)].set_index("query_id")
    acc = data.pivot(index="query_id", columns=["model", "harness"], values="acc")
    acc = acc.sort_index().sort_index(axis=1)
    routes = acc.columns.to_frame(index=False)
    scaffold = data.drop_duplicates(["model", "harness"]).set_index(["model", "harness"])
    routes["scaffold"] = scaffold.loc[acc.columns, "scaffold"].to_numpy()
    texts = queries["query"].loc[acc.index]
    return Benchmark(
        name=name,
        query_ids=acc.index.to_numpy(),
        texts=texts.tolist(),
        routes=routes,
        accuracy=acc.to_numpy(dtype=float),
        embeddings=query_embeddings(name, acc.index.to_numpy(), texts.tolist()),
    )


def query_embeddings(name: str, query_ids: np.ndarray, texts: list[str]) -> np.ndarray:
    """Qwen3-Embedding-8B query embeddings, cached in cache/embeddings/."""
    cache = CACHE / "embeddings" / f"{name}.npz"
    if cache.exists():
        with np.load(cache) as cached:
            assert (cached["query_id"] == query_ids.astype(str)).all()
            return cached["embeddings"]
    embeddings = encode(texts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, query_id=query_ids.astype(str), embeddings=embeddings)
    return embeddings


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def encode(texts: list[str]) -> np.ndarray:
    return embedding_model().encode(
        texts, batch_size=4, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def split_queries(size: int, seed: int, train_fraction: float = 0.7):
    order = np.random.default_rng(seed).permutation(size)
    split = round(train_fraction * size)
    return np.sort(order[:split]), np.sort(order[split:])
