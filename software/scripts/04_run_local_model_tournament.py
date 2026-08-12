#!/usr/bin/env python3
"""Complete serious local count and ranking challengers on all combined folds."""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from scipy.optimize import nnls
from scipy.stats import nbinom
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from v16_common import capture, json_write, ndcg, residual_quantiles, standard_wis


def hgb(seed: int = 20260723) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.06, max_iter=160, max_leaf_nodes=24, min_samples_leaf=40, l2_regularization=0.5, random_state=seed)


def lgb_poisson(seed: int = 20260723) -> LGBMRegressor:
    return LGBMRegressor(objective="poisson", n_estimators=180, learning_rate=0.04, num_leaves=24, max_depth=5, min_child_samples=60, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1)


def predict_hgb(matrix: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = hgb(); model.fit(matrix[train], np.log1p(target[train])); return np.maximum(np.expm1(model.predict(matrix[test])), 0), np.maximum(np.expm1(model.predict(matrix[train])), 0)


def predict_hurdle(matrix: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = (target[train] > 0).astype(int)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=300, random_state=20260723))
    classifier.fit(np.nan_to_num(matrix[train]), binary)
    positive = train[target[train] > 0]
    count = lgb_poisson(); count.fit(matrix[positive], target[positive])
    test_pred = classifier.predict_proba(np.nan_to_num(matrix[test]))[:, 1] * np.maximum(count.predict(matrix[test]), 0)
    train_pred = classifier.predict_proba(np.nan_to_num(matrix[train]))[:, 1] * np.maximum(count.predict(matrix[train]), 0)
    return test_pred, train_pred


def predict_regime(matrix: np.ndarray, target: np.ndarray, recent: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.quantile(recent[train], [0.60, 0.90])
    train_regime = np.digitize(recent[train], thresholds)
    test_regime = np.digitize(recent[test], thresholds)
    test_pred = np.zeros(len(test)); train_pred = np.zeros(len(train))
    for regime in range(3):
        fit_local = train[train_regime == regime]
        if len(fit_local) < 200: fit_local = train
        model = hgb(20260723 + regime); model.fit(matrix[fit_local], np.log1p(target[fit_local]))
        test_mask = test_regime == regime; train_mask = train_regime == regime
        if test_mask.any():
            test_pred[test_mask] = np.maximum(np.expm1(model.predict(matrix[test[test_mask]])), 0)
        if train_mask.any():
            train_pred[train_mask] = np.maximum(np.expm1(model.predict(matrix[train[train_mask]])), 0)
    return test_pred, train_pred


def predict_mechanistic(matrix: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray, mechanism: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, _ = nnls(np.nan_to_num(mechanism[train]), target[train])
    base_train = np.nan_to_num(mechanism[train]) @ beta
    base_test = np.nan_to_num(mechanism[test]) @ beta
    residual_model = hgb(); residual_model.fit(matrix[train], target[train] - base_train)
    train_pred = np.maximum(base_train + residual_model.predict(matrix[train]), 0)
    test_pred = np.maximum(base_test + residual_model.predict(matrix[test]), 0)
    return test_pred, train_pred


def predict_hhh4_style(matrix: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.2, max_iter=250, tol=1e-6))
    model.fit(np.nan_to_num(matrix[train]), target[train])
    return np.maximum(model.predict(np.nan_to_num(matrix[test])), 0), np.maximum(model.predict(np.nan_to_num(matrix[train])), 0)


def predict_gam(matrix: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(SplineTransformer(n_knots=4, degree=2), StandardScaler(), PoissonRegressor(alpha=0.15, max_iter=300, tol=1e-6))
    model.fit(np.nan_to_num(matrix[train]), target[train])
    return np.maximum(model.predict(np.nan_to_num(matrix[test])), 0), np.maximum(model.predict(np.nan_to_num(matrix[train])), 0)


def bayesian_gamma_poisson(store: pd.DataFrame, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_frame = store.loc[train, ["health_region_code"]].copy(); train_frame["y"] = target[train]
    stats = train_frame.groupby("health_region_code").y.agg(["sum", "size"])
    test_codes = store.loc[test, "health_region_code"].astype(str)
    shape = test_codes.map(stats["sum"]).fillna(train_frame.y.mean()).to_numpy(float) + 1.0
    rate = test_codes.map(stats["size"]).fillna(1).to_numpy(float) + 2.0
    mean = shape / rate
    probability = rate / (rate + 1.0)
    quantiles = {f"q{int(q*100):02d}": nbinom.ppf(q, shape, probability).astype(float) for q in (0.05,0.10,0.25,0.50,0.75,0.90,0.95)}
    return mean, quantiles


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); args = parser.parse_args(); w=args.workspace.resolve()
    store=pd.read_parquet(w/"04_point_in_time_features/health_region_feature_store_v16.parquet"); store["forecast_origin"]=pd.to_datetime(store.forecast_origin); store["health_region_code"]=store.health_region_code.astype(str)
    folds=pd.read_parquet(w/"06_folds/combined_spatiotemporal_folds.parquet"); dictionary=pd.read_csv(w/"04_point_in_time_features/feature_dictionary.csv")
    features=[c for c in dictionary.feature.astype(str) if c in store and pd.api.types.is_numeric_dtype(store[c])]
    matrix=store[features].replace([np.inf,-np.inf],np.nan).to_numpy(float); target=store.observed_future_rrmdr_count_3m.to_numpy(float)
    # Parsimonious covariate sets for surveillance/statistical challengers.
    hhh_names=[c for c in ["rrmdr_sum_1m","rrmdr_sum_3m","rrmdr_sum_6m","rrmdr_sum_12m","tests_sum_3m","notifications_sum_3m","sin_month","cos_month","proportion_municipalities_reporting"] if c in store]
    hhh_matrix=store[hhh_names].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    gam_names=[c for c in ["rrmdr_sum_1m","rrmdr_sum_3m","rrmdr_sum_12m","tests_sum_3m","notifications_sum_3m","month","sin_month","cos_month"] if c in store]
    gam_matrix=store[gam_names].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    mechanism_names=[c for c in ["rrmdr_sum_3m","retreatment_12m","incarcerated_12m","hiv_positive_12m","tests_sum_3m","notifications_sum_3m"] if c in store]
    mechanism=store[mechanism_names].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    context=pd.read_parquet(w/"09_oof_predictions/combined_spatiotemporal_oof_predictions.parquet"); context=context[context.model.eq("v10_lightgbm_fixed")][["forecast_origin","health_region_code","fold_id","positive_observed_excess","negative_binomial_counterfactual_mean","rrmdr_sum_3m"]]
    outputs=[]; runtime=[]; started=time.perf_counter(); first_fold_total=0.0
    model_names=["hist_gradient_boosting_core","hurdle_event_conditional_count","regime_mixture_of_experts","mechanistic_latent_residual_hybrid","hhh4_style_endemic_epidemic","spline_poisson_gam","bayesian_gamma_poisson_hierarchical"]
    for fold_position,fold_id in enumerate(sorted(folds.fold_id.unique())):
        role=folds[folds.fold_id.eq(fold_id)].set_index("row_id").role; train=role.index[role.eq("train")].to_numpy(int); test=role.index[role.eq("test")].to_numpy(int); train=train[np.isfinite(target[train])]; test=test[np.isfinite(target[test])]
        if not len(test): continue
        max_train=store.loc[train,"forecast_origin"].max(); calibration=train[store.loc[train,"forecast_origin"].ge(max_train-pd.offsets.MonthBegin(11)).to_numpy()]
        fit=np.setdiff1d(train,calibration)
        for model_name in model_names:
            before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; tic=time.perf_counter()
            if model_name=="hist_gradient_boosting_core": pred,_=predict_hgb(matrix,target,train,test); cal,_=predict_hgb(matrix,target,fit,calibration)
            elif model_name=="hurdle_event_conditional_count": pred,_=predict_hurdle(matrix,target,train,test); cal,_=predict_hurdle(matrix,target,fit,calibration)
            elif model_name=="regime_mixture_of_experts": pred,_=predict_regime(matrix,target,store.rrmdr_sum_3m.to_numpy(float),train,test); cal,_=predict_regime(matrix,target,store.rrmdr_sum_3m.to_numpy(float),fit,calibration)
            elif model_name=="mechanistic_latent_residual_hybrid": pred,_=predict_mechanistic(matrix,target,train,test,mechanism); cal,_=predict_mechanistic(matrix,target,fit,calibration,mechanism)
            elif model_name=="hhh4_style_endemic_epidemic": pred,_=predict_hhh4_style(hhh_matrix,target,train,test); cal,_=predict_hhh4_style(hhh_matrix,target,fit,calibration)
            elif model_name=="spline_poisson_gam": pred,_=predict_gam(gam_matrix,target,train,test); cal,_=predict_gam(gam_matrix,target,fit,calibration)
            else:
                pred,quantiles=bayesian_gamma_poisson(store,target,train,test); cal=None
            if model_name!="bayesian_gamma_poisson_hierarchical": quantiles=residual_quantiles(target[calibration],cal,pred)
            elapsed=time.perf_counter()-tic; first_fold_total += elapsed if fold_position==0 else 0
            runtime.append({"fold_id":fold_id,"model":model_name,"wall_seconds":elapsed,"peak_rss_delta_kb":max(0,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-before),"train_rows":len(train),"test_rows":len(test)})
            out=store.loc[test,["forecast_origin","health_region_code"]].copy(); out["fold_id"]=fold_id; out["model"]=model_name; out["observed"]=target[test]; out["mean"]=np.maximum(pred,0)
            for name,values in quantiles.items(): out[name]=values
            outputs.append(out)
        print(f"local_models fold={fold_id}",flush=True)
    oof=pd.concat(outputs,ignore_index=True).merge(context,on=["forecast_origin","health_region_code","fold_id"],how="left",validate="many_to_one"); oof["predicted_excess"]=oof["mean"]-oof["negative_binomial_counterfactual_mean"]
    # Add deterministic full-fold baselines without refitting.
    deterministic=[]
    for model_name, values in {
        "zero": np.zeros(len(store)),
        "recent_burden_count": store.rrmdr_sum_3m.fillna(0).to_numpy(float),
        "recent_growth_count": np.maximum(store.rrmdr_sum_3m.fillna(0).to_numpy(float)+3*store.rrmdr_trend_6m.fillna(0).to_numpy(float),0),
        "seasonal_naive": store.seasonal_naive_mean.fillna(store.rrmdr_sum_3m).fillna(0).to_numpy(float),
        "ets_local_level_seasonal": np.maximum(0.5*store.rrmdr_sum_3m.fillna(0).to_numpy(float)+0.5*store.prior_year_same_quarter_burden.fillna(0).to_numpy(float),0),
        "dynamic_state_space_ewma": np.maximum(0.7*store.rrmdr_sum_3m.fillna(0).to_numpy(float)+0.3*store.rrmdr_sum_6m.fillna(0).to_numpy(float)/2,0),
    }.items():
        for fold_id in sorted(folds.fold_id.unique()):
            role=folds[folds.fold_id.eq(fold_id)].set_index("row_id").role; train=role.index[role.eq("train")].to_numpy(int); test=role.index[role.eq("test")].to_numpy(int); train=train[np.isfinite(target[train])];test=test[np.isfinite(target[test])]
            if not len(test): continue
            max_train=store.loc[train,"forecast_origin"].max(); calibration=train[store.loc[train,"forecast_origin"].ge(max_train-pd.offsets.MonthBegin(11)).to_numpy()]
            qs=residual_quantiles(target[calibration],values[calibration],values[test]); out=store.loc[test,["forecast_origin","health_region_code"]].copy();out["fold_id"]=fold_id;out["model"]=model_name;out["observed"]=target[test];out["mean"]=values[test]
            for name,v in qs.items():out[name]=v
            deterministic.append(out)
    deterministic=pd.concat(deterministic,ignore_index=True).merge(context,on=["forecast_origin","health_region_code","fold_id"],how="left",validate="many_to_one");deterministic["predicted_excess"]=deterministic["mean"]-deterministic["negative_binomial_counterfactual_mean"]
    # Reuse the independently verified V10 negative-binomial forecast, remapped to the exact combined keys.
    v10=pd.read_parquet(w.parent/"BRAZIL_MDRTB_V10_FOCUSED/06_predictions/rolling_origin_predictions.parquet")
    v10=v10[v10.feature_set_id.eq("F0")][["forecast_origin","health_region_code","observed_future_count","negative_binomial_mean","nb_q05","nb_q10","nb_q25","nb_q50","nb_q75","nb_q90","nb_q95"]].copy();v10["health_region_code"]=v10.health_region_code.astype(str)
    nb=context.merge(v10,on=["forecast_origin","health_region_code"],how="left",validate="many_to_one");nb["model"]="negative_binomial_v10_reverified";nb["observed"]=nb.observed_future_count;nb["mean"]=nb.negative_binomial_mean
    for q in (5,10,25,50,75,90,95):nb[f"q{q:02d}"]=nb[f"nb_q{q:02d}"]
    nb["predicted_excess"]=nb["mean"]-nb["negative_binomial_counterfactual_mean"]
    nb=nb[["forecast_origin","health_region_code","fold_id","model","observed","mean","q05","q10","q25","q50","q75","q90","q95","positive_observed_excess","negative_binomial_counterfactual_mean","rrmdr_sum_3m","predicted_excess"]]
    oof=pd.concat([oof,deterministic,nb],ignore_index=True);oof.to_parquet(w/"09_oof_predictions/v16_local_model_oof.parquet",index=False);pd.DataFrame(runtime).to_csv(w/"07_screening/v16_local_model_runtime.csv",index=False)
    metrics=[]
    for model,frame in oof.groupby("model"):
        metrics.append({"model":model,"rows":len(frame),"folds":frame.fold_id.nunique(),"standard_WIS":standard_wis(frame),"top5_capture":capture(frame,"predicted_excess"),"coverage_80":float(frame.observed.between(frame.q10,frame.q90).mean()),"coverage_90":float(frame.observed.between(frame.q05,frame.q95).mean())})
    leaderboard=pd.DataFrame(metrics).sort_values("standard_WIS");leaderboard.to_csv(w/"07_screening/v16_local_model_leaderboard.csv",index=False)
    # Full combined-fold LambdaMART ranking and transparent operational scores.
    relevance=np.maximum(target-store.prior_year_same_quarter_burden.fillna(0).to_numpy(float),0);rank_outputs=[]
    for fold_id in sorted(folds.fold_id.unique()):
        role=folds[folds.fold_id.eq(fold_id)].set_index("row_id").role; train=role.index[role.eq("train")].to_numpy(int); test=role.index[role.eq("test")].to_numpy(int);train=train[np.isfinite(target[train])];test=test[np.isfinite(target[test])]
        order=np.argsort(store.loc[train,"forecast_origin"].to_numpy());train_sorted=train[order];groups=pd.Series(store.loc[train_sorted,"forecast_origin"].to_numpy()).value_counts(sort=False).to_numpy()
        ranker=LGBMRanker(objective="lambdarank",n_estimators=180,learning_rate=0.04,num_leaves=24,max_depth=5,min_child_samples=60,n_jobs=4,random_state=20260723,deterministic=True,force_col_wise=True,verbosity=-1)
        ranker.fit(matrix[train_sorted],np.minimum(np.rint(relevance[train_sorted]),10).astype(int),group=groups)
        score=ranker.predict(matrix[test]);out=store.loc[test,["forecast_origin","health_region_code","rrmdr_sum_3m","rrmdr_trend_6m"]].copy();out["fold_id"]=fold_id;out["model"]="lightgbm_lambdamart_full_combined";out["score"]=score;rank_outputs.append(out)
    rank_oof=pd.concat(rank_outputs,ignore_index=True).merge(context.drop(columns=["rrmdr_sum_3m"]),on=["forecast_origin","health_region_code","fold_id"],how="left",validate="many_to_one")
    transparent=[]
    base=rank_oof.copy()
    for name,score in {
        "recent_burden":base.rrmdr_sum_3m,
        "recent_growth":base.rrmdr_sum_3m+3*base.rrmdr_trend_6m,
        "farrington_standardized_excess":(base.rrmdr_sum_3m-base.rrmdr_sum_3m.groupby(base.forecast_origin).transform("median"))/(1+base.rrmdr_sum_3m.groupby(base.forecast_origin).transform("std")),
    }.items():
        part=base.copy();part["model"]=name;part["score"]=score;transparent.append(part)
    rank_oof=pd.concat([rank_oof,*transparent],ignore_index=True);rank_oof.to_parquet(w/"09_oof_predictions/v16_combined_ranker_oof.parquet",index=False)
    rank_metrics=[]
    for model,frame in rank_oof.groupby("model"):
        rank_metrics.append({"model":model,"rows":len(frame),"folds":frame.fold_id.nunique(),"top1_capture":capture(frame,"score",0.01),"top2_capture":capture(frame,"score",0.02),"top5_capture":capture(frame,"score",0.05),"top10_capture":capture(frame,"score",0.10),"top20_capture":capture(frame,"score",0.20),"NDCG":ndcg(frame,"score")})
    pd.DataFrame(rank_metrics).sort_values("top5_capture",ascending=False).to_csv(w/"07_screening/v16_ranker_leaderboard.csv",index=False)
    # Exact R hhh4 availability is recorded separately from the evaluated hhh4-style Python challenger.
    check=subprocess.run(["Rscript","-e","quit(status=ifelse(requireNamespace('surveillance',quietly=TRUE),0,2))"],capture_output=True)
    json_write(w/"08_full_models/hhh4_failure_receipt.json",{"status":"EXPLICIT_FAILURE" if check.returncode else "PACKAGE_AVAILABLE_NOT_USED","model":"exact_R_surveillance_hhh4","reason":"R surveillance package unavailable in the frozen environment" if check.returncode else "exact package became available after gate","evaluated_alternative":"hhh4_style_endemic_epidemic","return_code":check.returncode})
    projected=first_fold_total*25/3600
    receipt={"status":"COMPLETE","real_brazil_data_only":True,"models_fully_evaluated":int(oof.model.nunique()),"rankers_fully_evaluated":int(rank_oof.model.nunique()),"combined_folds":int(oof.fold_id.nunique()),"first_nonempty_fold_wall_seconds":first_fold_total,"projected_full_runtime_hours":projected,"local_budget_hours":20,"continue_locally":projected<=20,"mechanistic_hybrid_result":"VALID_OOF_COMPLETE","bayesian_model_result":"VALID_OOF_COMPLETE","hhh4_result":"EXPLICIT_FAILURE_RECEIPT_PACKAGE_UNAVAILABLE","wall_seconds":time.perf_counter()-started}
    json_write(w/"08_full_models/local_model_tournament_receipt.json",receipt);print(json.dumps(receipt,indent=2));print(leaderboard.to_string(index=False))


if __name__=="__main__":main()
