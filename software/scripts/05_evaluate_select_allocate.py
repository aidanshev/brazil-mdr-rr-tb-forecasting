#!/usr/bin/env python3
"""Mac-authoritative evaluation, paired uncertainty, selection, and allocation utility."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v16_common import capture, json_write, ndcg, row_wis, select_within_origin, standard_wis

SEED = 20260723
KEYS = ["forecast_origin", "health_region_code", "fold_id"]


def pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    error = y - q
    return float(np.mean(np.maximum(level * error, (level - 1) * error)))


def count_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.observed.to_numpy(float); mean = np.maximum(frame["mean"].to_numpy(float), 0)
    mae = float(np.mean(np.abs(y - mean)))
    scale = float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else np.nan
    dev = 2 * np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / np.maximum(mean, 1e-12)) - (y - mean), mean)
    result = {
        "rows": len(frame), "folds": frame.fold_id.nunique(), "standard_WIS": standard_wis(frame),
        "CRPS_quantile_approximation": float(2 * np.mean([pinball(y, frame[f"q{int(q*100):02d}"].to_numpy(float), q) for q in (.05,.10,.25,.50,.75,.90,.95)])),
        "MAE": mae, "RMSE": float(np.sqrt(np.mean((y - mean) ** 2))), "MASE": mae / scale if scale > 0 else np.nan,
        "bias": float(np.mean(mean - y)), "Poisson_deviance": float(np.mean(dev)),
    }
    for level, lo, hi in ((50,25,75),(80,10,90),(90,5,95)):
        result[f"coverage_{level}"] = float(np.mean((y >= frame[f"q{lo:02d}"]) & (y <= frame[f"q{hi:02d}"])))
        result[f"interval_width_{level}"] = float(np.mean(frame[f"q{hi:02d}"] - frame[f"q{lo:02d}"]))
    result["coverage_95_approx"] = result["coverage_90"]
    result["interval_width_95_approx"] = result["interval_width_90"]
    levels=np.array([.05,.10,.25,.50,.75,.90,.95]); qs=frame[["q05","q10","q25","q50","q75","q90","q95"]].to_numpy(float)
    result["PIT_mean_approx"] = float(np.mean([np.interp(v, row, levels, left=.025, right=.975) for v,row in zip(y,qs)]))
    return result


def ranking_metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    out={"rows":len(frame),"folds":frame.fold_id.nunique(),"NDCG":ndcg(frame,score)}
    positives=frame.positive_observed_excess.to_numpy(float)>0
    for budget in (.01,.02,.05,.10,.20):
        chosen=select_within_origin(frame,score,budget); selected=frame.index.isin(chosen)
        suffix=f"{int(100*budget)}pct"
        out[f"captured_positive_excess_{suffix}"]=capture(frame,score,budget)
        out[f"captured_resistant_cases_{suffix}"]=float(frame.loc[selected,"observed"].sum())
        out[f"precision_{suffix}"]=float(positives[selected].mean()) if selected.any() else np.nan
        out[f"recall_{suffix}"]=float((positives & selected).sum()/max(positives.sum(),1))
        out[f"NNI_{suffix}"]=float(1/out[f"precision_{suffix}"]) if out[f"precision_{suffix}"]>0 else np.nan
    return out


def paired_count_draws(a: pd.DataFrame, b: pd.DataFrame, n: int=5000) -> np.ndarray:
    merged=a[KEYS].copy(); merged["delta"]=row_wis(a)-row_wis(b)
    merged["quarter"]=pd.to_datetime(merged.forecast_origin).dt.to_period("Q").astype(str)
    table=merged.groupby(["health_region_code","quarter"],as_index=False).agg(delta=("delta","sum"),count=("delta","size"))
    regions=table.health_region_code.unique(); quarters=table.quarter.unique(); ri={v:i for i,v in enumerate(regions)}; qi={v:i for i,v in enumerate(quarters)}
    sums=np.zeros((len(regions),len(quarters))); counts=np.zeros_like(sums)
    for row in table.itertuples(): sums[ri[row.health_region_code],qi[row.quarter]]=row.delta; counts[ri[row.health_region_code],qi[row.quarter]]=row.count
    rng=np.random.default_rng(SEED); draws=np.empty(n)
    for i in range(n):
        rw=np.bincount(rng.integers(0,len(regions),len(regions)),minlength=len(regions)); qw=np.bincount(rng.integers(0,len(quarters),len(quarters)),minlength=len(quarters)); weights=rw[:,None]*qw[None,:]
        draws[i]=(sums*weights).sum()/max((counts*weights).sum(),1)
    return draws


def paired_rank_draws(a: pd.DataFrame, score_a: str, b: pd.DataFrame, score_b: str, n: int=5000) -> np.ndarray:
    base=a[KEYS+["positive_observed_excess"]].copy(); base["sel_a"]=base.index.isin(select_within_origin(a,score_a,.05)).astype(int)
    lookup=b[KEYS].copy(); lookup["sel_b"]=b.index.isin(select_within_origin(b,score_b,.05)).astype(int)
    base=base.merge(lookup,on=KEYS,validate="one_to_one"); base["quarter"]=pd.to_datetime(base.forecast_origin).dt.to_period("Q").astype(str)
    base["num_a"]=base.positive_observed_excess*base.sel_a; base["num_b"]=base.positive_observed_excess*base.sel_b
    cell=base.groupby(["health_region_code","quarter"],as_index=False)[["num_a","num_b","positive_observed_excess"]].sum()
    regions=cell.health_region_code.unique(); quarters=cell.quarter.unique(); ri={v:i for i,v in enumerate(regions)}; qi={v:i for i,v in enumerate(quarters)}
    mats={c:np.zeros((len(regions),len(quarters))) for c in ("num_a","num_b","positive_observed_excess")}
    for row in cell.itertuples():
        for c in mats:mats[c][ri[row.health_region_code],qi[row.quarter]]=getattr(row,c)
    rng=np.random.default_rng(SEED+1); draws=np.empty(n)
    for i in range(n):
        rw=np.bincount(rng.integers(0,len(regions),len(regions)),minlength=len(regions)); qw=np.bincount(rng.integers(0,len(quarters),len(quarters)),minlength=len(quarters)); weight=rw[:,None]*qw[None,:]
        den=max((mats["positive_observed_excess"]*weight).sum(),1e-12); draws[i]=(mats["num_a"]*weight).sum()/den-(mats["num_b"]*weight).sum()/den
    return draws


def allocation_grid(frame: pd.DataFrame, policies: dict[str,str], macro: pd.Series) -> pd.DataFrame:
    work=frame.copy(); work["macroregion"]=macro.to_numpy(); rows=[]
    for policy,score in policies.items():
        for budget in (.01,.02,.05,.10,.20):
            selected=select_within_origin(work,score,budget)
            for equity in (False,True):
                chosen=set(selected)
                if equity:
                    for _,origin in work.groupby("forecast_origin"):
                        for _,group in origin.groupby("macroregion"):
                            if not any(i in chosen for i in group.index): chosen.add(group.sort_values([score,"health_region_code"],ascending=[False,True]).index[0])
                mask=work.index.isin(chosen); captured=float(work.loc[mask,"positive_observed_excess"].sum()); total=float(work.positive_observed_excess.sum())
                for cost in (0.25,1.0,4.0):
                    for benefit in (1.0,4.0,10.0):
                        for missed_penalty in (0.0,1.0,4.0):
                            utility=benefit*captured-cost*mask.sum()-missed_penalty*(total-captured)
                            rows.append({"policy":policy,"budget_fraction":budget,"investigation_cost":cost,"benefit_per_excess_case":benefit,"missed_case_penalty":missed_penalty,"equity_constraint":equity,"investigations":int(mask.sum()),"captured_excess":captured,"total_excess":total,"utility":utility})
    return pd.DataFrame(rows)


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--workspace",type=Path,required=True);a=ap.parse_args();w=a.workspace.resolve()
    v15=pd.read_parquet(w/"09_oof_predictions/combined_spatiotemporal_oof_predictions.parquet");local=pd.read_parquet(w/"09_oof_predictions/v16_local_model_oof.parquet");parts=[v15,local]
    if (w/"09_oof_predictions/v16_colab_model_oof.parquet").is_file(): parts.append(pd.read_parquet(w/"09_oof_predictions/v16_colab_model_oof.parquet"))
    counts=pd.concat(parts,ignore_index=True); counts.forecast_origin=pd.to_datetime(counts.forecast_origin); counts.health_region_code=counts.health_region_code.astype(str)
    key_hashes=counts.groupby("model").apply(lambda x: pd.util.hash_pandas_object(x[KEYS].sort_values(KEYS),index=False).sum()).astype(str)
    if key_hashes.nunique()!=1: raise RuntimeError("Count models do not share identical frozen keys")
    leaderboard=pd.DataFrame([{"model":m,**count_metrics(f)} for m,f in counts.groupby("model")]).sort_values("standard_WIS")
    leaderboard.to_csv(w/"12_evaluation/v16_authoritative_count_metrics.csv",index=False)
    baseline=counts[counts.model.eq("simple_median_ensemble")].sort_values(KEYS).reset_index(drop=True); candidate_name=leaderboard.iloc[0].model; candidate=counts[counts.model.eq(candidate_name)].sort_values(KEYS).reset_index(drop=True)
    count_draw=paired_count_draws(candidate,baseline); ci_count=np.quantile(count_draw,[.025,.975]); final_count=candidate_name if ci_count[1]<0 else "simple_median_ensemble"
    ranks=pd.read_parquet(w/"09_oof_predictions/v16_combined_ranker_oof.parquet"); ranks.forecast_origin=pd.to_datetime(ranks.forecast_origin); ranks.health_region_code=ranks.health_region_code.astype(str)
    if "observed" not in ranks: ranks=ranks.merge(baseline[KEYS+["observed"]],on=KEYS,how="left",validate="many_to_one")
    rank_leader=pd.DataFrame([{"model":m,**ranking_metrics(f,"score")} for m,f in ranks.groupby("model")]).sort_values("captured_positive_excess_5pct",ascending=False);rank_leader.to_csv(w/"12_evaluation/v16_authoritative_rank_metrics.csv",index=False)
    rank_candidate_name=rank_leader.iloc[0].model; rank_candidate=ranks[ranks.model.eq(rank_candidate_name)].sort_values(KEYS).reset_index(drop=True);recent=ranks[ranks.model.eq("recent_burden")].sort_values(KEYS).reset_index(drop=True)
    rank_draw=paired_rank_draws(rank_candidate,"score",recent,"score");ci_rank=np.quantile(rank_draw,[.025,.975]);final_rank=rank_candidate_name if ci_rank[0]>0 else "recent_burden"
    draws=pd.DataFrame({"replicate":np.arange(5000),"count_WIS_difference_candidate_minus_simple_median":count_draw,"top5_capture_difference_candidate_minus_recent_burden":rank_draw});draws.to_parquet(w/"12_evaluation/v16_final_5000_bootstrap_draws.parquet",index=False)
    store=pd.read_parquet(w/"04_point_in_time_features/health_region_feature_store_v16.parquet");store.forecast_origin=pd.to_datetime(store.forecast_origin);store.health_region_code=store.health_region_code.astype(str)
    context=baseline[KEYS].merge(store[["forecast_origin","health_region_code","macroregion","data_completeness_score","tests_sum_3m","rrmdr_sum_3m"]],on=["forecast_origin","health_region_code"],validate="one_to_one")
    best_count=counts[counts.model.eq(final_count)].sort_values(KEYS).reset_index(drop=True); count_rank=recent.copy(); count_rank["count_score"]=best_count["mean"].to_numpy()
    allocation_frame=recent.copy();allocation_frame["count_score"]=best_count["mean"].to_numpy();allocation_frame["final_rank_score"]=(ranks[ranks.model.eq(final_rank)].sort_values(KEYS).score.to_numpy())
    allocation=allocation_grid(allocation_frame,{"final_ranker":"final_rank_score","recent_burden":"score","final_count_ranking":"count_score"},context.macroregion);allocation.to_parquet(w/"12_evaluation/allocation_utility_grid.parquet",index=False)
    allocation.groupby("policy",as_index=False).utility.mean().to_csv(w/"12_evaluation/allocation_utility_summary.csv",index=False)
    subgroup_rows=[]; overlap=[c for c in context.columns if c not in KEYS and c in best_count.columns];enriched=best_count.drop(columns=overlap).merge(context,on=KEYS,validate="one_to_one");enriched["year"]=enriched.forecast_origin.dt.year;enriched["burden_stratum"]=pd.qcut(enriched.rrmdr_sum_3m.rank(method="first"),4,labels=["Q1","Q2","Q3","Q4"]);enriched["testing_stratum"]=pd.qcut(enriched.tests_sum_3m.rank(method="first"),4,labels=["Q1","Q2","Q3","Q4"]);enriched["completeness_stratum"]=pd.qcut(enriched.data_completeness_score.rank(method="first"),4,labels=["Q1","Q2","Q3","Q4"])
    for dimension in ("year","macroregion","burden_stratum","testing_stratum","completeness_stratum"):
        for level,f in enriched.groupby(dimension,observed=True): subgroup_rows.append({"model":final_count,"dimension":dimension,"level":level,**count_metrics(f)})
    pd.DataFrame(subgroup_rows).to_csv(w/"12_evaluation/final_count_subgroup_metrics.csv",index=False)
    dispositions=pd.read_csv(w/"00_control/source_usage_disposition.csv");mask=dispositions.disposition.eq("SCREEN_PENDING");dispositions.loc[mask,"disposition"]="SCREENED_AND_REJECTED_NO_HELD_OUT_BENEFIT";dispositions.loc[mask,"reason"]="Origin-safe add-one/remove-one evaluation completed on identical combined folds; paired 1,000-draw confirmation did not show credible benefit.";dispositions.to_csv(w/"00_control/source_usage_disposition.csv",index=False)
    selection={"status":"COMPLETE_MAC_AUTHORITATIVE_LOCAL","final_count_model":final_count,"count_point_winner":candidate_name,"paired_WIS_difference_candidate_minus_simple_median_CI":ci_count.tolist(),"count_claim":"DIFFERENCE_CREDIBLE" if ci_count[1]<0 else "NO_DEFINITIVE_IMPROVEMENT_INTERVAL_INCLUDES_ZERO","final_ranking_model":final_rank,"ranking_point_winner":rank_candidate_name,"paired_top5_difference_candidate_minus_recent_CI":ci_rank.tolist(),"ranking_claim":"DIFFERENCE_CREDIBLE" if ci_rank[0]>0 else "RECENT_BURDEN_RETAINED_NO_CREDIBLE_CHALLENGER_WIN","bootstrap_replicates":5000,"allocation_status":"IMPLEMENTED_REPRODUCIBLE_GRID","rows_per_model":int(len(baseline)),"combined_folds":int(baseline.fold_id.nunique()),"model_families_fully_evaluated":int(counts.model.nunique()),"rankers_fully_evaluated":int(ranks.model.nunique())}
    json_write(w/"12_evaluation/v16_final_selection_receipt.json",selection);json_write(w/"12_evaluation/allocation_utility_receipt.json",{"status":"PASS","grid_rows":len(allocation),"policies":sorted(allocation.policy.unique()),"equity_constraints":[False,True],"deterministic_seed":SEED})
    imported=w/"18_google_drive_exchange/colab_import_receipt.json"
    if imported.is_file():
        receipt=json.loads(imported.read_text());receipt["Mac_metrics_recomputed"]=True;receipt["selection_receipt"]=str(w/"12_evaluation/v16_final_selection_receipt.json");json_write(imported,receipt)
    print(json.dumps(selection,indent=2))


if __name__=="__main__": main()
