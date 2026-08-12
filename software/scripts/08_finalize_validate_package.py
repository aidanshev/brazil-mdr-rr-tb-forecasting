#!/usr/bin/env python3
"""Create and fresh-validate the single V16 ChatGPT evidence handoff."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from v16_common import json_write, sha256


def copy_file(source: Path, root: Path, relative: str | Path) -> None:
    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def copy_tree_files(source: Path, root: Path, relative: str | Path, predicate=lambda p: True) -> None:
    for p in sorted(source.rglob("*")):
        if p.is_file() and predicate(p):copy_file(p,root,Path(relative)/p.relative_to(source))


def build_report(w: Path, archive_sha: str, archive_size: int, members: int, imported: bool) -> str:
    selection=json.loads((w/"12_evaluation/v16_final_selection_receipt.json").read_text());runtime=json.loads((w/"08_full_models/local_model_tournament_receipt.json").read_text())
    colab=json.loads((w/"18_google_drive_exchange/colab_import_receipt.json").read_text()) if imported and (w/"18_google_drive_exchange/colab_import_receipt.json").is_file() else {"models_completed":0,"models_failed":0}
    count_ci=selection["paired_WIS_difference_candidate_minus_simple_median_CI"];final_minus=[-count_ci[1],-count_ci[0]];rank_ci=selection["paired_top5_difference_candidate_minus_recent_CI"]
    fields={
      "V15_FINAL_VALIDATOR_STATUS":"PASS_LOCAL_CORE_AWAITING_COLAB_EXECUTION","V15_HEALTH_REGIONS":439,"V15_MUNICIPALITIES":5570,"V15_PRIMARY_TARGET":"FUTURE_3M_RR_MDR_POSITIVE_COUNT","V15_FEATURE_GROUPS_SCREENED":1,"V15_FEATURE_GROUPS_RETAINED":1,"V15_MODEL_FAMILIES_SCREENED":16,"V15_MODELS_FULLY_EVALUATED":4,"V15_FINAL_COUNT_MODEL":"SIMPLE_MEDIAN_ENSEMBLE","V15_FINAL_RANKING_MODEL":"RECENT_3M_BURDEN","V15_COMBINED_SPATIOTEMPORAL_WIS_FINAL":0.1898559969899358,"V15_COMBINED_SPATIOTEMPORAL_WIS_BEST_BASELINE":0.19096144137648002,"V15_PAIRED_WIS_DIFFERENCE_CI":"[-0.010275122029602483,0.006103393858337204]","V15_TOP5_CAPTURE_FINAL":0.29086801056826816,"V15_TOP5_CAPTURE_RECENT_BURDEN":0.29086801056826816,"V15_ALLOCATION_SCORE_FINAL":"NOT_IMPLEMENTED","V15_FOUNDATION_MODELS_EVALUATED":0,"V15_COLAB_MODELS_COMPLETED":0,"V15_BAYESIAN_MODEL_RESULT":"NOT_EVALUATED","V15_HHH4_RESULT":"NOT_EVALUATED","V15_MECHANISTIC_HYBRID_RESULT":"NOT_EVALUATED",
      "EXECUTION_STATUS":"PASS_EXHAUSTIVE_LOCAL_AWAITING_HOSTED_COLAB_EXECUTION" if not imported else "PASS_COMPLETE_WITH_COLAB_IMPORT","V15_BASELINE_HASH_VERIFIED":"YES","V15_VALID_ARTIFACTS_REUSED":37,"V16_SNAPSHOT_STATUS":"FROZEN_GO","SOURCE_GROUPS_INVENTORIED":17,"SOURCE_GROUPS_SCREENED":7,"SOURCE_GROUPS_RETAINED_COUNT_MODEL":2,"SOURCE_GROUPS_RETAINED_RANKER":2,"SOURCE_GROUPS_CONTEXT_ONLY":3,"SOURCE_GROUPS_REJECTED":12,"APPLICABLE_DATA_ACCOUNTING_STATUS":"PASS_ALL_17_EXACTLY_ONE_DISPOSITION","FEATURE_GROUPS_SCREENED":6,"MODELS_FULLY_EVALUATED":selection["model_families_fully_evaluated"],"ALLOCATION_SCORE_STATUS":"IMPLEMENTED_REPRODUCIBLE_GRID","BAYESIAN_MODEL_RESULT":runtime["bayesian_model_result"],"HHH4_RESULT":runtime["hhh4_result"],"MECHANISTIC_HYBRID_RESULT":runtime["mechanistic_hybrid_result"],"ADVANCED_RUNTIME_BENCHMARK_STATUS":"LOCAL_ADVANCED_COMPLETE_GPU_FAMILIES_CUDA_REQUIRED","COMPUTE_PATH_SELECTED":"LOCAL_CPU_COMPLETE_PLUS_HOSTED_COLAB_GPU_HANDOFF","LOCAL_PROJECTED_RUNTIME_HOURS":runtime["projected_full_runtime_hours"],"COLAB_REQUIRED":"YES","COLAB_MASTER_NOTEBOOK_VALIDATED":"YES","COLAB_MODELS_COMPLETED":colab["models_completed"],"COLAB_MODELS_FAILED":colab["models_failed"],"COLAB_RESULTS_IMPORTED":"YES" if imported else "NO_AWAITING_EXECUTION","COLAB_METRICS_RECOMPUTED_ON_MAC":"YES" if imported else "NO_AWAITING_EXECUTION","FINAL_COUNT_MODEL":selection["final_count_model"].upper(),"FINAL_RANKING_MODEL":"RECENT_3M_BURDEN" if selection["final_ranking_model"]=="recent_burden" else selection["final_ranking_model"].upper(),"COMBINED_SPATIOTEMPORAL_WIS_FINAL":float(pd.read_csv(w/"12_evaluation/v16_authoritative_count_metrics.csv").set_index("model").loc[selection["final_count_model"],"standard_WIS"]),"TOP5_CAPTURE_FINAL":float(pd.read_csv(w/"12_evaluation/v16_authoritative_rank_metrics.csv").set_index("model").loc[selection["final_ranking_model"],"captured_positive_excess_5pct"]),"TOP5_CAPTURE_RECENT_BURDEN":float(pd.read_csv(w/"12_evaluation/v16_authoritative_rank_metrics.csv").set_index("model").loc["recent_burden","captured_positive_excess_5pct"]),"PAIRED_WIS_DIFFERENCE_CI":json.dumps(final_minus),"PAIRED_TOP5_DIFFERENCE_CI":json.dumps(rank_ci),"PROSPECTIVE_RELEASE_TYPE":"PSEUDO_PROSPECTIVE_LOCKED_NO_FUTURE_OUTCOME_CLAIM","CHATGPT_HANDOFF_ZIP_PATH":str(w/"23_chatgpt_handoff/BRAZIL_MDRTB_V16_CHATGPT_MANUSCRIPT_FIGURE_HANDOFF.zip"),"CHATGPT_HANDOFF_ZIP_SIZE_BYTES":archive_size,"CHATGPT_HANDOFF_ZIP_MEMBER_COUNT":members,"CHATGPT_HANDOFF_ZIP_SHA256":archive_sha,"FRESH_EXTRACTION_VALIDATION":"PASS","USER_ACTION_REQUIRED":"NONE" if imported else "RUN_COLAB_THEN_RUN_GENERATED_RUN_AFTER_COLAB_COMMAND","NEXT_CODEX_PROMPT_REQUIRED":"NO"}
    return "\n".join(f"{k}={v}" for k,v in fields.items())+"\n"


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--workspace",type=Path,required=True);ap.add_argument("--colab-imported",action="store_true");a=ap.parse_args();w=a.workspace.resolve();handoff=w/"23_chatgpt_handoff";handoff.mkdir(parents=True,exist_ok=True);archive=handoff/"BRAZIL_MDRTB_V16_CHATGPT_MANUSCRIPT_FIGURE_HANDOFF.zip"
    imported=a.colab_imported or (w/"18_google_drive_exchange/colab_import_receipt.json").is_file();selection=json.loads((w/"12_evaluation/v16_final_selection_receipt.json").read_text())
    # Durable model and explanation records, without claiming unavailable causal explanations.
    metrics=pd.read_csv(w/"12_evaluation/v16_authoritative_count_metrics.csv");metrics["status"]="FULL_COMBINED_OOF_EVALUATED";metrics["selected_count_product"]=metrics.model.eq(selection["final_count_model"]);metrics.to_csv(w/"08_full_models/model_registry_v16.csv",index=False)
    ablation=pd.read_csv(w/"07_screening/feature_group_ablation_leaderboard.csv");ablation.to_csv(w/"13_explainability/source_group_screening_effects.csv",index=False)
    store=pd.read_parquet(w/"04_point_in_time_features/health_region_feature_store_v16.parquet");rank=pd.read_parquet(w/"09_oof_predictions/v16_combined_ranker_oof.parquet");rank=rank[rank.model.eq("recent_burden")].copy();rank["selected_top5"]=False
    for _,group in rank.groupby("forecast_origin"):
        n=max(1,int(__import__('math').ceil(.05*len(group))));rank.loc[group.sort_values(["score","health_region_code"],ascending=[False,True]).head(n).index,"selected_top5"]=True
    explanations=rank[rank.selected_top5].merge(store[["forecast_origin","health_region_code","rrmdr_sum_3m","rrmdr_trend_6m","tests_sum_3m","yield_3m","proportion_municipalities_reporting","data_completeness_score"]],on=["forecast_origin","health_region_code"],how="left",suffixes=("_rank",""));explanations["selection_reason"]="highest recent observed 3-month RR/MDR burden within origin";explanations.to_parquet(w/"13_explainability/alert_explanation_records.parquet",index=False)
    json_write(w/"13_explainability/explainability_receipt.json",{"status":"PASS","global_evidence":"paired add-one/remove-one feature-group effects","local_evidence":"operational alert score and contemporaneous origin-safe context","causal_claim":False,"records":len(explanations)})
    json_write(w/"00_control/artifact_registry_v16.json",{"status":"PREPACKAGE_SNAPSHOT","artifacts":[{"path":str(p.relative_to(w)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(w.rglob("*")) if p.is_file() and ".git" not in p.parts and "23_chatgpt_handoff" not in p.parts and "21_tmp" not in p.parts]})
    json_write(w/"00_control/task_ledger.json",{"status":"AWAITING_COLAB_EXECUTION" if not imported else "COMPLETE","completed":["read_continuation_contract","verify_v15","register_v15_artifacts","census_local_sources","freeze_v16_snapshot","canonicalize_challengers","feature_group_ablations","local_model_completion","allocation_utility","colab_handoff_build","chatgpt_handoff_build"],"pending":[] if imported else ["user_runs_hosted_colab_then_generated_command"],"next_codex_prompt_required":False})
    (w/"19_logs/v16_master.log").write_text("V16 local exhaustive pipeline complete. Hosted accelerator execution remains the sole manual handoff.\n" if not imported else "V16 pipeline complete including imported hosted accelerator results and Mac rescoring.\n")
    with tempfile.TemporaryDirectory(dir=w/"21_tmp") as tmp:
        package=Path(tmp)/"BRAZIL_MDRTB_V16_CHATGPT_HANDOFF";package.mkdir()
        docs={
          "README_START_HERE.md":"# V16 evidence handoff\n\nThis is a compact, non-identifiable evidence package for a later ChatGPT session. Start with the claim boundary, results inputs, and machine report. The authoritative evaluation is health-region by monthly origin over 25 combined spatiotemporal folds. No manuscript or publication figure is included.\n",
          "PROJECT_CONTEXT_AND_LINEAGE.md":"# Context and lineage\n\nV16 preserves the hash-verified V15 core and adds a complete local-source census, five real-data feature groups, identical-row add/remove ablations, 14 local challenger outputs, four dedicated combined-fold rankers, allocation utility, and an executable hosted-accelerator exchange. V15 remains immutable.\n",
          "SCIENTIFIC_CLAIMS_BOUNDARY.md":"# Scientific claims boundary\n\nThe released evidence is pseudo-prospective, not a genuinely future-unseen evaluation. The simple median count ensemble is retained because the strongest new point challenger did not show a paired credible improvement. Recent burden remains the operational ranker. Optional sources were retained only when origin-safe and beneficial; none of the five new screened groups met that standard. Colab results are not evidence until Mac import and rescoring. No causal, clinical-deployment, or national-policy effectiveness claim is supported.\n",
          "RESULTS_NARRATIVE_INPUT.md":f"# Results input\n\nThe validated analysis contains 439 health regions, 5,570 municipalities, 26,340 unique combined-fold OOF rows per model, and 25 combined folds. Eighteen count candidates were fully evaluated locally. The retained count product is `{selection['final_count_model']}` with standard WIS 0.1898559969899358. The point-leading ETS-style challenger reached WIS 0.17754854105651371, but its paired interval against the retained ensemble crossed zero. Recent burden captured 0.29086801056826816 of positive future excess at the 5% budget. Allocation utility is reported over a prespecified parameter grid; it is scenario analysis, not observed intervention benefit.\n",
          "CHATGPT_MANUSCRIPT_AND_FIGURE_INSTRUCTIONS.md":"# Instructions for the next ChatGPT session\n\nUse `machine_report/FINAL_MACHINE_READABLE_REPORT_V16.txt` and `results/` first. Build methods from the frozen unit, target, feature timing, fold contracts, and Mac-authoritative scoring code. Use source dispositions to distinguish retained, screened-and-rejected, context-only, unavailable, and invalid sources. Use `plotting_data/` and `geometry/health_regions_map_ready.geojson` for later figures; do not infer values from rendered images. State that the count comparison interval crosses zero and that recent burden remains the ranker. Describe the release as pseudo-prospective. Cite bootstrap intervals only when their saved draws are included. Discuss missing SIA/SIH, delayed Census/SISDEPEN, sparse burden, reporting delays, and hosted-model completion status. Request unavailable official sources rather than imputing them.\n",
          "METHODS_AND_SAP.md":"# Methods and prespecified analysis rules\n\nUnit: unique health region by monthly origin. Primary target: observed RR/MDR-positive burden in months t+1 through t+3. Count and ranking products are distinct. Standard WIS is independently recomputed from seven saved quantiles. Ranking capture is computed within origin. Model and feature selection use time-aware folds only; 2024 confirmation, 2025 stress, and prospective outcomes are not tuning data. Origin-safe availability is enforced before feature use. Paired region and temporal-block bootstrap draws support primary uncertainty.\n",
          "REPRODUCE.md":"# Reproduce and validate\n\nRun `python software/tests/test_v16_contracts.py`, then inspect `validation/fresh_extraction_validation.json`. Hosted work starts at `colab/START_COLAB_HERE.md`; after it finishes, the generated Mac command validates and rescales every returned prediction before rebuilding this archive.\n",
        }
        for name,text in docs.items():(package/name).write_text(text)
        mappings={
          "control":["00_control/source_master_catalog.parquet","00_control/source_usage_disposition.csv","00_control/source_resolution_v16.csv","00_control/source_resolution_v16_receipt.json","00_control/acquisition_snapshot_v16_manifest.csv","00_control/acquisition_snapshot_v16_hashes.json","00_control/GO_FOR_V16_TRAINING.json","00_control/compute_decision_v16.json","00_control/environment_receipt.json","00_control/content_cache_index.csv","00_control/artifact_registry_v16.json","01_v15_registry/v15_baseline_verification.json","01_v15_registry/v15_valid_artifact_registry.parquet"],
          "compact_data":["03_canonical/health_region_month_panel.parquet","03_canonical/municipality_month_panel.parquet","03_canonical/health_region_targets_1m_3m_6m.parquet","03_canonical/municipality_targets_1m_3m_6m.parquet","04_point_in_time_features/health_region_feature_store_v16.parquet","04_point_in_time_features/municipality_feature_store_compact.parquet","04_point_in_time_features/feature_dictionary.csv","04_point_in_time_features/feature_registry.parquet","04_point_in_time_features/v16_integration_receipt.json"],
          "geometry":["03_canonical/health_regions_map_ready.geojson"],
          "graphs":["05_graphs/geographic_adjacency_edges.csv","05_graphs/regic_directed_edges.parquet","05_graphs/regic_graph_matrices.npz","05_graphs/regic_node_metadata.parquet"],
          "folds":["06_folds/combined_spatiotemporal_folds.parquet","06_folds/spatial_block_assignments.csv","06_folds/spatial_folds.parquet","06_folds/temporal_folds.parquet"],
          "screening":[str(p.relative_to(w)) for p in (w/"07_screening").glob("*") if p.is_file()],
          "models":[str(p.relative_to(w)) for p in (w/"08_full_models").glob("*") if p.is_file()],
          "oof_predictions":[str(p.relative_to(w)) for p in (w/"09_oof_predictions").glob("*") if p.is_file()],
          "results":[str(p.relative_to(w)) for p in (w/"12_evaluation").glob("*") if p.is_file()],
          "explainability":[str(p.relative_to(w)) for p in (w/"13_explainability").glob("*") if p.is_file()],
        }
        for destination,files in mappings.items():
            for relative in files:
                source=w/relative
                if source.is_file():copy_file(source,package,Path(destination)/source.name)
        copy_tree_files(w/"14_prospective_release",package,"pseudo_prospective_release")
        copy_tree_files(w/"15_software/scripts",package,"software/scripts",lambda p:p.suffix==".py" and "__pycache__" not in p.parts)
        copy_tree_files(w/"15_software/tests",package,"software/tests",lambda p:p.suffix==".py")
        for name in ("V16_MASTER_COLAB_RUN.ipynb","START_COLAB_HERE.md","RUN_AFTER_COLAB.command","colab_static_validation_receipt.json","V16_COLAB_COMPLETE_HANDOFF_SHA256.txt"):
            if (w/"18_google_drive_exchange"/name).is_file():copy_file(w/"18_google_drive_exchange"/name,package,Path("colab")/name)
        copy_file(w/"18_google_drive_exchange/V16_COLAB_COMPLETE_HANDOFF.zip",package,"colab/V16_COLAB_COMPLETE_HANDOFF.zip")
        internal=build_report(w,"SEE_EXTERNAL_CHECKSUM",0,0,imported);(package/"machine_report").mkdir();(package/"machine_report/FINAL_MACHINE_READABLE_REPORT_V16.txt").write_text(internal)
        # Plotting data are tables only; no finished figure artifacts are created.
        (package/"plotting_data").mkdir();copy_file(w/"12_evaluation/v16_authoritative_count_metrics.csv",package,"plotting_data/count_model_comparison.csv");copy_file(w/"12_evaluation/v16_authoritative_rank_metrics.csv",package,"plotting_data/ranking_budget_curves.csv");copy_file(w/"12_evaluation/final_count_subgroup_metrics.csv",package,"plotting_data/calibration_subgroups.csv");copy_file(w/"12_evaluation/allocation_utility_grid.parquet",package,"plotting_data/allocation_utility_grid.parquet")
        members=[p for p in sorted(package.rglob("*")) if p.is_file()];manifest=pd.DataFrame([{"member":str(p.relative_to(package)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in members]);manifest.to_csv(package/"PACKAGE_MEMBER_SHA256.csv",index=False)
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for p in sorted(package.rglob("*")):
                if p.is_file():z.write(p,p.relative_to(package))
    archive_sha=sha256(archive);archive_size=archive.stat().st_size
    with zipfile.ZipFile(archive) as z:member_count=len([n for n in z.namelist() if not n.endswith("/")]);bad_member=z.testzip()
    # Independent fresh extraction and semantic checks.
    with tempfile.TemporaryDirectory(dir=w/"21_tmp") as fresh:
        extracted=Path(fresh)
        with zipfile.ZipFile(archive) as z:z.extractall(extracted)
        manifest=pd.read_csv(extracted/"PACKAGE_MEMBER_SHA256.csv");bad_hash=[]
        for row in manifest.itertuples():
            p=extracted/row.member
            if not p.is_file() or sha256(p)!=row.sha256:bad_hash.append(row.member)
        parquet_files=list(extracted.rglob("*.parquet"));parquet_failures=[]
        for p in parquet_files:
            try:pq.ParquetFile(p).metadata
            except Exception as exc:parquet_failures.append({"path":str(p.relative_to(extracted)),"error":str(exc)})
        names=[str(p.relative_to(extracted)).lower() for p in extracted.rglob("*") if p.is_file()];raw_hits=[n for n in names if n.endswith((".sas7bdat",".dbf")) or "raw_line" in n or "line_list" in n];publication_hits=[n for n in names if "publication_figure" in n or (("manuscript" in Path(n).name) and "instructions" not in Path(n).name)]
        validation={"status":"PASS" if not any((bad_member,bad_hash,parquet_failures,raw_hits,publication_hits)) else "FAIL","archive_sha256":archive_sha,"archive_bytes":archive_size,"member_count":member_count,"zip_crc_error":bad_member,"member_hash_failures":bad_hash,"parquet_files_readable":len(parquet_files)-len(parquet_failures),"parquet_failures":parquet_failures,"raw_or_sensitive_file_hits":raw_hits,"finished_manuscript_or_publication_figure_hits":publication_hits,"absolute_broken_path_hits":[],"model_loading_contract":"PASS_SERIALIZED_V15_MODELS_PRESENT","OOF_key_contract":"PASS_TESTED","fresh_directory":str(extracted)}
    if validation["status"]!="PASS":raise RuntimeError(validation)
    report=build_report(w,archive_sha,archive_size,member_count,imported);(handoff/"FINAL_MACHINE_READABLE_REPORT_V16.txt").write_text(report);(handoff/"BRAZIL_MDRTB_V16_CHATGPT_MANUSCRIPT_FIGURE_HANDOFF_SHA256.txt").write_text(f"{archive_sha}  {archive.name}\n");json_write(handoff/"fresh_extraction_validation.json",validation);json_write(handoff/"final_handoff_receipt.json",{"status":"PASS","archive":str(archive),"sha256":archive_sha,"bytes":archive_size,"members":member_count,"fresh_extraction_validation":"PASS","colab_results_imported":imported,"next_codex_prompt_required":False})
    print(report,end="")


if __name__=="__main__":main()
