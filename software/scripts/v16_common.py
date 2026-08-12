#!/usr/bin/env python3
"""Shared deterministic helpers for the V16 continuation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
SEED = 20260723


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def stable_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].sort_values(columns).astype(str)
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def row_wis(frame: pd.DataFrame) -> np.ndarray:
    y = frame["observed"].to_numpy(float)
    total = 0.5 * np.abs(y - frame["q50"].to_numpy(float))
    for alpha, lower, upper in ((0.5, 25, 75), (0.2, 10, 90), (0.1, 5, 95)):
        lo = frame[f"q{lower:02d}"].to_numpy(float)
        hi = frame[f"q{upper:02d}"].to_numpy(float)
        total += (alpha / 2) * (
            hi - lo
            + (2 / alpha) * (lo - y) * (y < lo)
            + (2 / alpha) * (y - hi) * (y > hi)
        )
    return total / 3.5


def standard_wis(frame: pd.DataFrame) -> float:
    return float(np.nanmean(row_wis(frame)))


def residual_quantiles(
    train_y: np.ndarray, train_prediction: np.ndarray, future_prediction: np.ndarray
) -> dict[str, np.ndarray]:
    residual = np.asarray(train_y, float) - np.asarray(train_prediction, float)
    values = np.sort(
        np.vstack(
            [np.maximum(np.asarray(future_prediction, float) + np.nanquantile(residual, q), 0) for q in QUANTILES]
        ),
        axis=0,
    )
    return {f"q{int(q * 100):02d}": values[i] for i, q in enumerate(QUANTILES)}


def select_within_origin(frame: pd.DataFrame, score: str, budget: float) -> pd.Index:
    selected: list[object] = []
    for _, origin in frame.groupby("forecast_origin", sort=True):
        n = max(1, math.ceil(float(budget) * len(origin)))
        selected.extend(
            origin.sort_values([score, "health_region_code"], ascending=[False, True]).head(n).index
        )
    return pd.Index(selected)


def capture(frame: pd.DataFrame, score: str, budget: float = 0.05) -> float:
    denominator = float(frame["positive_observed_excess"].sum())
    if denominator <= 0:
        return float("nan")
    return float(frame.loc[select_within_origin(frame, score, budget), "positive_observed_excess"].sum() / denominator)


def ndcg(frame: pd.DataFrame, score: str) -> float:
    values: list[float] = []
    for _, origin in frame.groupby("forecast_origin", sort=True):
        relevance = origin["positive_observed_excess"].to_numpy(float)
        order = np.argsort(origin[score].to_numpy(float))[::-1]
        ideal = np.argsort(relevance)[::-1]
        discount = 1 / np.log2(np.arange(2, len(origin) + 2))
        dcg = float(np.sum((2 ** relevance[order] - 1) * discount))
        idcg = float(np.sum((2 ** relevance[ideal] - 1) * discount))
        values.append(dcg / idcg if idcg > 0 else np.nan)
    return float(np.nanmean(values))


def environment_receipt() -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "statsmodels", "duckdb", "geopandas"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"UNAVAILABLE:{type(exc).__name__}"
    try:
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except Exception:
        git_branch = "UNKNOWN"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "git_branch": git_branch,
        "cuda_available": False,
        "authoritative_compute": "Mac/external-drive CPU",
    }
