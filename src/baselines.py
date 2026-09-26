"""Baselines: Random and Avengers (arXiv:2505.19797)."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans

CLUSTERS = 16


def random_accuracy(accuracy: np.ndarray, test: np.ndarray) -> float:
    """Expected accuracy of picking a route uniformly at random."""
    return float(accuracy[test].mean())


def avengers_select(
    features: np.ndarray, accuracy: np.ndarray, train: np.ndarray,
    test: np.ndarray, seed: int, clusters: int = CLUSTERS,
) -> np.ndarray:
    """k-means the train queries; route each test query to the best route
    of its nearest (cosine) cluster."""
    clusters = min(clusters, len(train))
    kmeans = KMeans(n_clusters=clusters, init="k-means++", n_init=10,
                    max_iter=1_000, random_state=seed)
    labels = kmeans.fit_predict(features[train])
    centers = kmeans.cluster_centers_
    profiles = np.stack(
        [accuracy[train][labels == c].mean(axis=0) for c in range(clusters)]
    )

    queries = features[test]
    norms = np.linalg.norm(queries, axis=1, keepdims=True) * np.linalg.norm(
        centers, axis=1, keepdims=True
    ).T
    similarity = np.divide(queries @ centers.T, norms,
                           out=np.zeros((len(queries), clusters)), where=norms > 0)
    nearest = np.argsort(-similarity, axis=1)[:, 0]
    return profiles[nearest].argmax(axis=1)
