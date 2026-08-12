#!/usr/bin/env python3
"""Run real-data Stage 1 CPU screening with a two-fold runtime gate."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def wis(frame: pd.DataFrame) -> np.ndarray:
    observed = frame["observed"].to_numpy(float)
    total = 0.5 * np.abs(observed - frame["q50"].to_numpy(float))
    for alpha, lower, upper in ((0.5, 25, 75), (0.2, 10, 90), (0.1, 5, 95)):
        lo = frame[f"q{lower:02d}"].to_numpy(float)
        hi = frame[f"q{upper:02d}"].to_numpy(float)
        total += (alpha / 2) * (
            hi - lo
            + (2 / alpha) * (lo - observed) * (observed < lo)
            + (2 / alpha) * (observed - hi) * (observed > hi)
        )
    return total / 3.5


def capture(frame: pd.DataFrame, score: str) -> float:
    selected = []
    for _, origin in frame.groupby("forecast_origin", sort=True):
        count = max(1, math.ceil(0.05 * len(origin)))
        selected.extend(
            origin.sort_values([score, "health_region_code"], ascending=[False, True]).head(count).index
        )
    denominator = frame["positive_observed_excess"].sum()
    return float(frame.loc[selected, "positive_observed_excess"].sum() / denominator)


def model_factories() -> dict[str, callable]:
    return {
        "hist_gradient_boosting_log": lambda: HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.06, max_iter=180, max_leaf_nodes=24,
            min_samples_leaf=40, l2_regularization=0.5, random_state=20260723,
        ),
        "extra_trees_log": lambda: ExtraTreesRegressor(
            n_estimators=240, min_samples_leaf=8, max_features=0.7, n_jobs=4, random_state=20260723,
        ),
        "poisson_glm": lambda: make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=300, tol=1e-7)
        ),
        "elastic_net_log": lambda: make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.1, max_iter=5000, random_state=20260723)
        ),
    }


def predict(model_name: str, model: object, matrix: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict(matrix), float)
    if model_name.endswith("_log"):
        raw = np.expm1(raw)
    return np.maximum(raw, 0)


def croston(history: np.ndarray, alpha: float = 0.1) -> float:
    nonzero = np.flatnonzero(history > 0)
    if not len(nonzero):
        return 0.0
    demand = float(history[nonzero[0]])
    interval = float(nonzero[0] + 1)
    previous = nonzero[0]
    for index in nonzero[1:]:
        demand += alpha * (float(history[index]) - demand)
        interval += alpha * (float(index - previous) - interval)
        previous = index
    return demand / max(interval, 1e-9)


def intermittent_forecasts(panel: pd.DataFrame, origins: set[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for code, series in panel.groupby("health_region_code", sort=True):
        series = series.sort_values("period_start")
        values = series["rrmdr_positive_cases"].to_numpy(float)
        dates = pd.to_datetime(series["period_start"]).tolist()
        for index, origin in enumerate(dates):
            if origin not in origins:
                continue
            history = values[: index + 1]
            base = croston(history)
            sba = 0.95 * base
            aggregate_3 = np.array([history[max(0, len(history) - 3 * (j + 1)): len(history) - 3 * j].sum() for j in range(max(1, len(history) // 3))])
            adida = float(np.mean(aggregate_3)) if len(aggregate_3) else 0.0
            imapa_components = []
            for level in (1, 3, 6):
                usable = len(history) // level
                if usable:
                    aggregated = history[-usable * level:].reshape(usable, level).sum(axis=1)
                    imapa_components.append(croston(aggregated) / level)
            imapa = float(np.mean(imapa_components)) if imapa_components else 0.0
            nonzero_probability = float(pd.Series(history).ewm(alpha=0.1, adjust=False).mean().iloc[-1] > 0)
            tsb = nonzero_probability * float(history[history > 0].mean()) if np.any(history > 0) else 0.0
            for model, monthly in (("croston", base), ("sba", sba), ("tsb", tsb), ("adida", adida / 3), ("imapa", imapa)):
                rows.append(
                    {"health_region_code": str(code), "forecast_origin": origin, "model": model, "mean": max(0.0, monthly * 3)}
                )
    return pd.DataFrame(rows)


def add_residual_quantiles(train_observed: np.ndarray, train_predictions: np.ndarray, test_predictions: np.ndarray) -> dict[str, np.ndarray]:
    residual = train_observed - train_predictions
    matrix = np.vstack([np.maximum(test_predictions + np.quantile(residual, q), 0) for q in QUANTILES])
    matrix = np.sort(matrix, axis=0)
    return {f"q{int(q * 100):02d}": matrix[index] for index, q in enumerate(QUANTILES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--v10-root", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    screen_dir = workspace / "07_screening"
    prediction_dir = workspace / "09_oof_predictions"
    screen_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    store = pd.read_parquet(workspace / "04_point_in_time_features/health_region_feature_store.parquet")
    folds = pd.read_parquet(workspace / "06_folds/temporal_folds.parquet")
    dictionary = pd.read_csv(workspace / "04_point_in_time_features/feature_dictionary.csv")
    features = [name for name in dictionary["feature"] if name in store and pd.api.types.is_numeric_dtype(store[name])]
    feature_matrix = store[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    target = store["observed_future_rrmdr_count_3m"].to_numpy(float)
    v10_oof = pd.read_parquet(args.v10_root / "06_predictions/rolling_origin_predictions.parquet")
    v10_oof = v10_oof[v10_oof["feature_set_id"].eq("F0")].copy()
    v10_oof["health_region_code"] = v10_oof["health_region_code"].astype(str)
    context = v10_oof[
        ["forecast_origin", "health_region_code", "positive_observed_excess", "negative_binomial_counterfactual_mean"]
    ].copy()

    outputs = []
    runtime_rows = []
    factories = model_factories()
    fold_ids = [f"temporal_{year}" for year in range(2020, 2025)]
    screening_start = time.perf_counter()
    first_two_seconds = 0.0
    for fold_position, fold_id in enumerate(fold_ids):
        fold = folds[folds["fold_id"].eq(fold_id)].set_index("row_id")
        train_ids = fold.index[fold["role"].eq("train")].to_numpy(int)
        test_ids = fold.index[fold["role"].eq("test")].to_numpy(int)
        train_ids = train_ids[np.isfinite(target[train_ids])]
        test_ids = test_ids[np.isfinite(target[test_ids])]
        calibration_cut = pd.to_datetime(store.loc[train_ids, "forecast_origin"]).max() - pd.offsets.MonthBegin(11)
        calibration_ids = train_ids[pd.to_datetime(store.loc[train_ids, "forecast_origin"]).ge(calibration_cut)]
        fit_ids = np.setdiff1d(train_ids, calibration_ids, assume_unique=False)
        for model_name, factory in factories.items():
            before_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            started = time.perf_counter()
            calibrator = factory()
            calibrator.fit(feature_matrix[fit_ids], np.log1p(target[fit_ids]) if model_name.endswith("_log") else target[fit_ids])
            calibration_prediction = predict(model_name, calibrator, feature_matrix[calibration_ids])
            model = factory()
            model.fit(feature_matrix[train_ids], np.log1p(target[train_ids]) if model_name.endswith("_log") else target[train_ids])
            mean = predict(model_name, model, feature_matrix[test_ids])
            quantiles = add_residual_quantiles(target[calibration_ids], calibration_prediction, mean)
            elapsed = time.perf_counter() - started
            after_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            runtime_rows.append(
                {
                    "fold_id": fold_id,
                    "model": model_name,
                    "wall_seconds": elapsed,
                    "peak_rss_delta_kb": max(0, after_memory - before_memory),
                    "train_rows": len(train_ids),
                    "test_rows": len(test_ids),
                }
            )
            if fold_position < 2:
                first_two_seconds += elapsed
            prediction = store.loc[test_ids, ["forecast_origin", "health_region_code"]].copy()
            prediction["health_region_code"] = prediction["health_region_code"].astype(str)
            prediction["fold_id"] = fold_id
            prediction["model"] = model_name
            prediction["observed"] = target[test_ids]
            prediction["mean"] = mean
            for name, values in quantiles.items():
                prediction[name] = values
            outputs.append(prediction)
        if fold_position == 1:
            projected = first_two_seconds * 5 / 2
            if projected > 20 * 3600:
                break

    cpu_oof = pd.concat(outputs, ignore_index=True)
    all_origins = {pd.Timestamp(value) for value in pd.to_datetime(cpu_oof["forecast_origin"]).unique()}
    panel = pd.read_parquet(workspace / "03_canonical/health_region_month_panel.parquet")
    intermittent = intermittent_forecasts(panel, all_origins)
    observed_keys = store[
        ["forecast_origin", "health_region_code", "observed_future_rrmdr_count_3m"]
    ].copy()
    observed_keys["health_region_code"] = observed_keys["health_region_code"].astype(str)
    intermittent = intermittent.merge(observed_keys, on=["forecast_origin", "health_region_code"], how="left", validate="many_to_one")
    for model_name, model_frame in intermittent.groupby("model"):
        for fold_id in fold_ids:
            year = int(fold_id.rsplit("_", 1)[1])
            test = model_frame[pd.to_datetime(model_frame["forecast_origin"]).dt.year.eq(year)].copy()
            history = model_frame[pd.to_datetime(model_frame["forecast_origin"]).dt.year.lt(year)].dropna(subset=["observed_future_rrmdr_count_3m"])
            if test.empty or history.empty:
                continue
            quantiles = add_residual_quantiles(
                history["observed_future_rrmdr_count_3m"].to_numpy(float), history["mean"].to_numpy(float), test["mean"].to_numpy(float)
            )
            test["fold_id"] = fold_id
            test["observed"] = test["observed_future_rrmdr_count_3m"]
            for name, values in quantiles.items():
                test[name] = values
            outputs.append(test[["forecast_origin", "health_region_code", "fold_id", "model", "observed", "mean", *[f"q{int(q*100):02d}" for q in QUANTILES]]])

    oof = pd.concat(outputs, ignore_index=True)
    oof = oof.merge(context, on=["forecast_origin", "health_region_code"], how="left", validate="many_to_one")
    oof["predicted_excess"] = oof["mean"] - oof["negative_binomial_counterfactual_mean"]
    oof.to_parquet(prediction_dir / "stage_1_cpu_oof_predictions.parquet", index=False)
    metrics = []
    for model, frame in oof.groupby("model"):
        metrics.append(
            {
                "model": model,
                "rows": len(frame),
                "folds": frame["fold_id"].nunique(),
                "standard_WIS": float(wis(frame).mean()),
                "MAE": float(np.abs(frame["observed"] - frame["mean"]).mean()),
                "top5_captured_positive_excess": capture(frame, "predicted_excess"),
            }
        )
    leaderboard = pd.DataFrame(metrics).sort_values("standard_WIS")
    leaderboard.to_csv(screen_dir / "stage_1_cpu_leaderboard.csv", index=False)
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(screen_dir / "stage_1_runtime_benchmark.csv", index=False)
    projected_seconds = first_two_seconds * 5 / 2
    receipt = {
        "status": "COMPLETE" if oof["fold_id"].nunique() == 5 else "PRUNED_BY_RUNTIME_GATE",
        "real_frozen_data": True,
        "first_two_fold_wall_seconds": first_two_seconds,
        "projected_full_cpu_model_seconds": projected_seconds,
        "projected_full_cpu_model_hours": projected_seconds / 3600,
        "local_budget_hours": 20,
        "continue_locally": projected_seconds <= 20 * 3600,
        "models": sorted(oof["model"].unique()),
        "features": len(features),
        "total_wall_seconds": time.perf_counter() - screening_start,
        "missing_required_dependencies": ["xgboost", "catboost"],
        "note": "V10 LightGBM, seasonal naive, and negative-binomial baselines are retained in the independently corrected V10 audit leaderboard.",
    }
    (screen_dir / "stage_1_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
