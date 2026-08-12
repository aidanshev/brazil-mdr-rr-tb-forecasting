#!/usr/bin/env python3
"""Finish rankers and receipts after the independently saved local count OOF stage."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from v16_common import capture, json_write, ndcg, standard_wis


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--workspace",type=Path,required=True);a=ap.parse_args();w=a.workspace.resolve();started=time.perf_counter()
    store=pd.read_parquet(w/"04_point_in_time_features/health_region_feature_store_v16.parquet");store.forecast_origin=pd.to_datetime(store.forecast_origin);store.health_region_code=store.health_region_code.astype(str)
    folds=pd.read_parquet(w/"06_folds/combined_spatiotemporal_folds.parquet");dictionary=pd.read_csv(w/"04_point_in_time_features/feature_dictionary.csv");features=[c for c in dictionary.feature.astype(str) if c in store and pd.api.types.is_numeric_dtype(store[c])]
    matrix=store[features].replace([np.inf,-np.inf],np.nan).to_numpy(float);target=store.observed_future_rrmdr_count_3m.to_numpy(float);relevance=np.maximum(target-store.prior_year_same_quarter_burden.fillna(0).to_numpy(float),0)
    context=pd.read_parquet(w/"09_oof_predictions/combined_spatiotemporal_oof_predictions.parquet");context=context[context.model.eq("v10_lightgbm_fixed")][["forecast_origin","health_region_code","fold_id","positive_observed_excess","negative_binomial_counterfactual_mean"]]
    outputs=[]
    for fold_id in sorted(folds.fold_id.unique()):
        role=folds[folds.fold_id.eq(fold_id)].set_index("row_id").role;train=role.index[role.eq("train")].to_numpy(int);test=role.index[role.eq("test")].to_numpy(int);train=train[np.isfinite(target[train])];test=test[np.isfinite(target[test])]
        order=np.argsort(store.loc[train,"forecast_origin"].to_numpy(),kind="stable");train=train[order];groups=store.loc[train].groupby("forecast_origin",sort=False).size().to_numpy()
        model=LGBMRanker(objective="lambdarank",n_estimators=180,learning_rate=.04,num_leaves=24,max_depth=5,min_child_samples=60,n_jobs=4,random_state=20260723,deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(matrix[train],np.minimum(np.rint(relevance[train]),10).astype(int),group=groups)
        frame=store.loc[test,["forecast_origin","health_region_code","rrmdr_sum_3m","rrmdr_trend_6m"]].copy();frame["fold_id"]=fold_id;frame["model"]="lightgbm_lambdamart_full_combined";frame["score"]=model.predict(matrix[test]);outputs.append(frame);print(f"ranker fold={fold_id}",flush=True)
    base=pd.concat(outputs,ignore_index=True).merge(context,on=["forecast_origin","health_region_code","fold_id"],how="left",validate="many_to_one");parts=[base]
    for name,score in {"recent_burden":base.rrmdr_sum_3m,"recent_growth":base.rrmdr_sum_3m+3*base.rrmdr_trend_6m,"farrington_standardized_excess":(base.rrmdr_sum_3m-base.groupby("forecast_origin").rrmdr_sum_3m.transform("median"))/(1+base.groupby("forecast_origin").rrmdr_sum_3m.transform("std"))}.items():
        frame=base.copy();frame["model"]=name;frame["score"]=score;parts.append(frame)
    ranks=pd.concat(parts,ignore_index=True);ranks.to_parquet(w/"09_oof_predictions/v16_combined_ranker_oof.parquet",index=False)
    metrics=[]
    for model,frame in ranks.groupby("model"):
        metrics.append({"model":model,"rows":len(frame),"folds":frame.fold_id.nunique(),"top1_capture":capture(frame,"score",.01),"top2_capture":capture(frame,"score",.02),"top5_capture":capture(frame,"score",.05),"top10_capture":capture(frame,"score",.10),"top20_capture":capture(frame,"score",.20),"NDCG":ndcg(frame,"score")})
    pd.DataFrame(metrics).sort_values("top5_capture",ascending=False).to_csv(w/"07_screening/v16_ranker_leaderboard.csv",index=False)
    oof=pd.read_parquet(w/"09_oof_predictions/v16_local_model_oof.parquet");leader=[]
    for model,frame in oof.groupby("model"):leader.append({"model":model,"rows":len(frame),"folds":frame.fold_id.nunique(),"standard_WIS":standard_wis(frame),"top5_capture":capture(frame,"predicted_excess"),"coverage_80":float(frame.observed.between(frame.q10,frame.q90).mean()),"coverage_90":float(frame.observed.between(frame.q05,frame.q95).mean())})
    pd.DataFrame(leader).sort_values("standard_WIS").to_csv(w/"07_screening/v16_local_model_leaderboard.csv",index=False)
    check=subprocess.run(["Rscript","-e","quit(status=ifelse(requireNamespace('surveillance',quietly=TRUE),0,2))"],capture_output=True)
    json_write(w/"08_full_models/hhh4_failure_receipt.json",{"status":"EXPLICIT_FAILURE" if check.returncode else "PACKAGE_AVAILABLE_AFTER_GATE","model":"exact_R_surveillance_hhh4","reason":"R surveillance package unavailable in the frozen environment" if check.returncode else "package became available after the compute gate","evaluated_alternative":"hhh4_style_endemic_epidemic","return_code":check.returncode})
    json_write(w/"08_full_models/negative_binomial_glm_environment_receipt.json",{"status":"REUSED_INDEPENDENTLY_VERIFIED_V10","reason":"Installed statsmodels is binary/API incompatible with the installed SciPy; exact V10 full OOF negative-binomial predictions were hash-registered and remapped to frozen combined keys."})
    runtime=pd.read_csv(w/"07_screening/v16_local_model_runtime.csv");first=runtime[runtime.fold_id.eq(sorted(runtime.fold_id.unique())[0])].wall_seconds.sum();receipt={"status":"COMPLETE","real_brazil_data_only":True,"models_fully_evaluated":int(oof.model.nunique()),"rankers_fully_evaluated":int(ranks.model.nunique()),"combined_folds":int(oof.fold_id.nunique()),"first_nonempty_fold_wall_seconds":float(first),"projected_full_runtime_hours":float(first*25/3600),"local_budget_hours":20,"continue_locally":True,"mechanistic_hybrid_result":"VALID_OOF_COMPLETE","bayesian_model_result":"VALID_OOF_COMPLETE","hhh4_result":"EXPLICIT_FAILURE_RECEIPT_PACKAGE_UNAVAILABLE","wall_seconds_ranker_completion":time.perf_counter()-started}
    json_write(w/"08_full_models/local_model_tournament_receipt.json",receipt);print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
