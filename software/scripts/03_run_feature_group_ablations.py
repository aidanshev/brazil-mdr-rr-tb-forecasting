#!/usr/bin/env python3
"""Run identical-row add-one/remove-one feature-group ablations on all 25 combined folds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from v16_common import capture, json_write, residual_quantiles, row_wis, standard_wis


def factory(seed: int = 20260723) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1", n_estimators=144, learning_rate=0.056255784149979234,
        num_leaves=25, max_depth=4, min_child_samples=93, subsample=0.8453277029519232,
        colsample_bytree=0.8383802201148424, reg_alpha=0.32513304913597635,
        reg_lambda=0.006531111417298453, min_split_gain=0.13688017911150818,
        max_bin=255, n_jobs=4, random_state=seed, deterministic=True, force_col_wise=True,
        verbosity=-1,
    )


def fit_predict(matrix: np.ndarray, target: np.ndarray, train_ids: np.ndarray, test_ids: np.ndarray, origins: pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    max_train = pd.to_datetime(origins.iloc[train_ids]).max()
    calibration_ids = train_ids[pd.to_datetime(origins.iloc[train_ids]).ge(max_train - pd.offsets.MonthBegin(11))]
    fit_ids = np.setdiff1d(train_ids, calibration_ids)
    calibration_model = factory()
    calibration_model.fit(matrix[fit_ids], np.log1p(target[fit_ids]))
    calibration_prediction = np.maximum(np.expm1(calibration_model.predict(matrix[calibration_ids])), 0)
    model = factory()
    model.fit(matrix[train_ids], np.log1p(target[train_ids]))
    prediction = np.maximum(np.expm1(model.predict(matrix[test_ids])), 0)
    return prediction, residual_quantiles(target[calibration_ids], calibration_prediction, prediction)


def block_bootstrap(base: pd.DataFrame, challenger: pd.DataFrame, replicates: int) -> pd.DataFrame:
    keys = ["forecast_origin", "health_region_code", "fold_id"]
    base = base.sort_values(keys).reset_index(drop=True).copy()
    challenger = challenger.sort_values(keys).reset_index(drop=True).copy()
    assert base[keys].equals(challenger[keys])
    base["wis"] = row_wis(base)
    challenger["wis"] = row_wis(challenger)
    base["rank_score"] = base["predicted_excess"]
    challenger["rank_score"] = challenger["predicted_excess"]
    region = base["health_region_code"].astype(str)
    quarter = pd.to_datetime(base["forecast_origin"]).dt.to_period("Q").astype(str)
    regions = np.array(sorted(region.unique()))
    quarters = np.array(sorted(quarter.unique()))
    rng = np.random.default_rng(20260723)
    rows = []
    for bootstrap_id in range(replicates):
        region_count = pd.Series(rng.choice(regions, len(regions), replace=True)).value_counts()
        quarter_count = pd.Series(rng.choice(quarters, len(quarters), replace=True)).value_counts()
        weights = region.map(region_count).fillna(0).to_numpy(int) * quarter.map(quarter_count).fillna(0).to_numpy(int)
        indices = np.repeat(np.arange(len(base)), weights)
        if not len(indices): continue
        b, c = base.iloc[indices].copy(), challenger.iloc[indices].copy()
        rows.extend([
            {"bootstrap_id": bootstrap_id, "metric": "WIS", "estimate_base": float(b.wis.mean()), "estimate_challenger": float(c.wis.mean()), "paired_difference": float(c.wis.mean() - b.wis.mean())},
            {"bootstrap_id": bootstrap_id, "metric": "top5_capture", "estimate_base": capture(b, "rank_score"), "estimate_challenger": capture(c, "rank_score"), "paired_difference": capture(c, "rank_score") - capture(b, "rank_score")},
        ])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    w = args.workspace.resolve()
    store = pd.read_parquet(w / "04_point_in_time_features/health_region_feature_store_v16.parquet")
    store["forecast_origin"] = pd.to_datetime(store["forecast_origin"])
    folds = pd.read_parquet(w / "06_folds/combined_spatiotemporal_folds.parquet")
    base_dictionary = pd.read_csv(w / "04_point_in_time_features/feature_dictionary.csv")
    base_features = [c for c in base_dictionary["feature"].astype(str) if c in store and pd.api.types.is_numeric_dtype(store[c])]
    groups: dict[str, list[str]] = json.loads((w / "04_point_in_time_features/v16_feature_groups.json").read_text())
    groups = {g: [c for c in cols if c in store and pd.api.types.is_numeric_dtype(store[c])] for g, cols in groups.items()}
    all_added = [c for group in sorted(groups) for c in groups[group]]
    configurations: dict[str, list[str]] = {"F0_core": base_features}
    configurations.update({f"F0_plus_{group}": base_features + groups[group] for group in sorted(groups)})
    configurations["FULL_origin_safe"] = base_features + all_added
    configurations.update({f"FULL_minus_{group}": base_features + [c for g in sorted(groups) if g != group for c in groups[g]] for group in sorted(groups)})

    target = store["observed_future_rrmdr_count_3m"].to_numpy(float)
    context = pd.read_parquet(w / "09_oof_predictions/combined_spatiotemporal_oof_predictions.parquet")
    context = context[context["model"].eq("v10_lightgbm_fixed")][["forecast_origin", "health_region_code", "fold_id", "positive_observed_excess", "negative_binomial_counterfactual_mean", "rrmdr_sum_3m"]].copy()
    context["health_region_code"] = context["health_region_code"].astype(str)
    outputs = []
    runtimes = []
    started = time.perf_counter()
    fold_ids = sorted(folds.fold_id.unique())
    for config_id, features in configurations.items():
        matrix = store[features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        config_start = time.perf_counter()
        for fold_position, fold_id in enumerate(fold_ids):
            role = folds[folds.fold_id.eq(fold_id)].set_index("row_id")["role"]
            train_ids = role.index[role.eq("train")].to_numpy(int)
            test_ids = role.index[role.eq("test")].to_numpy(int)
            train_ids = train_ids[np.isfinite(target[train_ids])]
            test_ids = test_ids[np.isfinite(target[test_ids])]
            if not len(test_ids): continue
            fold_start = time.perf_counter()
            prediction, quantiles = fit_predict(matrix, target, train_ids, test_ids, store["forecast_origin"])
            output = store.loc[test_ids, ["forecast_origin", "health_region_code"]].copy()
            output["health_region_code"] = output.health_region_code.astype(str)
            output["fold_id"] = fold_id
            output["configuration"] = config_id
            output["observed"] = target[test_ids]
            output["mean"] = prediction
            for name, values in quantiles.items(): output[name] = values
            outputs.append(output)
            runtimes.append({"configuration": config_id, "fold_id": fold_id, "wall_seconds": time.perf_counter() - fold_start, "train_rows": len(train_ids), "test_rows": len(test_ids), "features": len(features)})
        print(f"ablation={config_id} seconds={time.perf_counter()-config_start:.2f}", flush=True)
    oof = pd.concat(outputs, ignore_index=True).merge(context, on=["forecast_origin", "health_region_code", "fold_id"], how="left", validate="many_to_one")
    oof["predicted_excess"] = oof["mean"] - oof["negative_binomial_counterfactual_mean"]
    oof.to_parquet(w / "09_oof_predictions/feature_group_ablation_oof.parquet", index=False)
    pd.DataFrame(runtimes).to_csv(w / "07_screening/feature_group_ablation_runtime.csv", index=False)
    metrics = []
    for config_id, frame in oof.groupby("configuration"):
        metrics.append({"configuration": config_id, "rows": len(frame), "folds": frame.fold_id.nunique(), "features": len(configurations[config_id]), "standard_WIS": standard_wis(frame), "top5_capture": capture(frame.assign(rank_score=frame.predicted_excess), "rank_score"), "coverage_80": float(frame.observed.between(frame.q10, frame.q90).mean()), "coverage_90": float(frame.observed.between(frame.q05, frame.q95).mean())})
    metrics_frame = pd.DataFrame(metrics).sort_values("standard_WIS")
    metrics_frame.to_csv(w / "07_screening/feature_group_ablation_leaderboard.csv", index=False)

    base = oof[oof.configuration.eq("F0_core")]
    draws = []
    summaries = []
    retained_count, retained_ranker = [], []
    for group in sorted(groups):
        challenger = oof[oof.configuration.eq(f"F0_plus_{group}")]
        draw = block_bootstrap(base, challenger, args.replicates)
        draw["feature_group"] = group
        draws.append(draw)
        for metric, values in draw.groupby("metric"):
            low, high = values.paired_difference.quantile([0.025, 0.975])
            point = float(values.paired_difference.mean())
            summaries.append({"feature_group": group, "metric": metric, "paired_difference": point, "ci_low": low, "ci_high": high, "replicates": len(values)})
            if metric == "WIS" and point < 0 and high < 0: retained_count.append(group)
            if metric == "top5_capture" and point > 0 and low > 0: retained_ranker.append(group)
    draw_frame = pd.concat(draws, ignore_index=True)
    draw_frame.to_parquet(w / "12_evaluation/feature_group_ablation_bootstrap_draws.parquet", index=False)
    pd.DataFrame(summaries).to_csv(w / "12_evaluation/feature_group_ablation_bootstrap_summary.csv", index=False)
    selection = {"status": "PASS", "screened_groups": sorted(groups), "retained_count_groups": retained_count, "retained_ranker_groups": retained_ranker, "retention_rule": "paired 95% block-bootstrap interval must exclude zero in beneficial direction", "identical_rows": True, "identical_fold_keys": True, "outer_rows_per_configuration": int(base.shape[0]), "combined_folds": int(base.fold_id.nunique()), "bootstrap_replicates_per_group_metric": args.replicates, "wall_seconds": time.perf_counter() - started}
    json_write(w / "07_screening/feature_group_selection_receipt.json", selection)
    # Stage summaries show the prespecified successive-halving domains from the same saved OOF rows.
    stage_rows = []
    for config_id, frame in oof.groupby("configuration"):
        temporal_year = frame.fold_id.str.extract(r"temporal_(\d{4})")[0].astype(int)
        spatial_block = frame.fold_id.str.extract(r"spatial_(\d+)")[0].astype(int)
        for stage, mask in [("first_two_temporal_folds", temporal_year.le(2021)), ("all_five_temporal_folds", temporal_year.le(2024)), ("spatial_validation", spatial_block.notna()), ("combined_spatiotemporal_confirmation", pd.Series(True, index=frame.index))]:
            subset = frame[mask.to_numpy()]
            stage_rows.append({"configuration": config_id, "stage": stage, "rows": len(subset), "folds": subset.fold_id.nunique(), "standard_WIS": standard_wis(subset), "top5_capture": capture(subset.assign(rank_score=subset.predicted_excess), "rank_score")})
    pd.DataFrame(stage_rows).to_csv(w / "07_screening/successive_halving_stage_metrics.csv", index=False)
    print(json.dumps(selection, indent=2))
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
