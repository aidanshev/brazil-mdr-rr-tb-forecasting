#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
import zipfile
from pathlib import Path

import pandas as pd


W=Path(__file__).resolve().parents[2]
ROOT=W.parent
sys.path.insert(0,str(W/"15_software/scripts"))
from v16_common import select_within_origin
KEYS=["forecast_origin","health_region_code","fold_id"]
ALLOWED={"RETAINED_IN_FINAL_MODEL","RETAINED_IN_FINAL_RANKER","RETAINED_FOR_PROSPECTIVE_CONTEXT_ONLY","SCREENED_AND_REJECTED_NO_HELD_OUT_BENEFIT","EXCLUDED_NOT_ORIGIN_SAFE","EXCLUDED_INSUFFICIENT_TEMPORAL_COVERAGE","EXCLUDED_INVALID_OR_JUNK","EXCLUDED_DUPLICATE_OR_SUPERSEDED","REQUEST_REQUIRED_NOT_AVAILABLE"}


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


class V16Contracts(unittest.TestCase):
    def test_every_source_has_one_allowed_disposition(self):
        d=pd.read_csv(W/"00_control/source_usage_disposition.csv")
        self.assertFalse(d.source_group.duplicated().any());self.assertTrue(set(d.disposition)<=ALLOWED);self.assertEqual(len(d),17)

    def test_all_valid_origin_safe_groups_screened_or_retained(self):
        d=pd.read_csv(W/"00_control/source_usage_disposition.csv")
        for group in ("SINAN_TB","GEOGRAPHY_POPULATION","CNES","SIM_MORTALITY","APS_PMMB_PREVINE_UBS","REGIC_MOBILITY","CLIMATE"):
            self.assertIn(d.set_index("source_group").loc[group,"disposition"],{"RETAINED_IN_FINAL_MODEL","RETAINED_IN_FINAL_RANKER","SCREENED_AND_REJECTED_NO_HELD_OUT_BENEFIT"})

    def test_v15_archive_unchanged_and_v16_snapshot_frozen(self):
        archive=ROOT/"BRAZIL_MDRTB_V15_FINAL_REAL_PREDICTIVE_PLATFORM/22_final_package/BRAZIL_MDRTB_V15_FINAL_REAL_PREDICTIVE_PLATFORM.zip"
        self.assertEqual(digest(archive),"fd91325e649a581c2a2464f00f5f1c90c2b3f76f996cca451cb7c2aef9370402")
        gate=json.loads((W/"00_control/GO_FOR_V16_TRAINING.json").read_text());self.assertEqual(gate["status"],"GO")

    def test_paired_ablations_have_identical_rows(self):
        x=pd.read_parquet(W/"09_oof_predictions/feature_group_ablation_oof.parquet")
        counts=x.groupby("configuration").size();self.assertEqual(counts.nunique(),1);self.assertEqual(int(counts.iloc[0]),26340)

    def test_available_time_does_not_exceed_origin(self):
        store=pd.read_parquet(W/"04_point_in_time_features/health_region_feature_store_v16.parquet");store.forecast_origin=pd.to_datetime(store.forecast_origin)
        mapping={"F1_cnes_capacity.parquet":"f1_cnes_any_source_available","F4_sim_mortality.parquet":"f4_source_available","F5_pmmb_primary_care.parquet":"f5_source_available","F8_regic_connectivity.parquet":"f8_source_available","F9_inmet_climate.parquet":"f9_source_available"}
        for filename,indicator in mapping.items():
            canonical=pd.read_parquet(W/"03_canonical"/filename,columns=["available_time"]);first=pd.to_datetime(canonical.available_time).min();early=store[store.forecast_origin<first]
            if len(early):self.assertTrue(early[indicator].fillna(0).eq(0).all())
        dictionary=pd.read_csv(W/"04_point_in_time_features/feature_dictionary.csv");self.assertTrue(dictionary.origin_safe.astype(bool).all())
        d=pd.read_csv(W/"00_control/source_usage_disposition.csv").set_index("feature_group");self.assertEqual(d.loc["F7","disposition"],"RETAINED_FOR_PROSPECTIVE_CONTEXT_ONLY")

    def test_all_oof_models_match_frozen_keys(self):
        x=pd.read_parquet(W/"09_oof_predictions/v16_local_model_oof.parquet");sizes=x.groupby("model").size();self.assertEqual(sizes.nunique(),1);self.assertEqual(int(sizes.iloc[0]),26340)
        self.assertTrue((x.groupby("model").fold_id.nunique()==25).all())

    def test_source_specific_canonical_contracts(self):
        required={"entity_id","municipality_code","health_region_code","month","feature_time","available_time","source_vintage","source_file","source_sha256","transform_version","missingness_reason"}
        for name in ("F1_cnes_capacity.parquet","F4_sim_mortality.parquet","F5_pmmb_primary_care.parquet","F8_regic_connectivity.parquet","F9_inmet_climate.parquet"):
            self.assertTrue(required<=set(pd.read_parquet(W/"03_canonical"/name).columns))

    def test_ranking_budget_is_within_each_origin(self):
        x=pd.read_parquet(W/"09_oof_predictions/v16_combined_ranker_oof.parquet");x=x[x.model.eq("recent_burden")]
        chosen=x.loc[select_within_origin(x,"score",.05)]
        selected_counts=chosen.groupby("forecast_origin").size()
        for origin,g in x.groupby("forecast_origin"):
            self.assertLessEqual(int(selected_counts.get(origin,0)),max(1,math.ceil(.05*len(g))))

    def test_allocation_reproducible(self):
        r=json.loads((W/"12_evaluation/allocation_utility_receipt.json").read_text());self.assertEqual(r["status"],"PASS");self.assertEqual(r["deterministic_seed"],20260723)

    def test_advanced_local_results_or_failure_receipts(self):
        r=json.loads((W/"08_full_models/local_model_tournament_receipt.json").read_text());self.assertEqual(r["bayesian_model_result"],"VALID_OOF_COMPLETE");self.assertEqual(r["mechanistic_hybrid_result"],"VALID_OOF_COMPLETE");self.assertTrue((W/"08_full_models/hhh4_failure_receipt.json").is_file())

    def test_colab_notebook_has_complete_static_contract(self):
        text=(W/"18_google_drive_exchange/V16_MASTER_COLAB_RUN.ipynb").read_text().lower()
        for marker in ("todo","notimplementederror","codex should","placeholder","smoke","fake","unresolved model revision"):
            self.assertNotIn(marker,text)
        receipt=json.loads((W/"18_google_drive_exchange/colab_static_validation_receipt.json").read_text());self.assertEqual(receipt["status"],"PASS")

    def test_colab_state_is_honest(self):
        imported=W/"18_google_drive_exchange/colab_import_receipt.json"
        if imported.is_file(): self.assertTrue(json.loads(imported.read_text())["Mac_metrics_recomputed"])
        else: self.assertTrue((W/"18_google_drive_exchange/V16_COLAB_COMPLETE_HANDOFF.zip").is_file())

    def test_final_handoff_if_present(self):
        archive=W/"23_chatgpt_handoff/BRAZIL_MDRTB_V16_CHATGPT_MANUSCRIPT_FIGURE_HANDOFF.zip"
        if not archive.is_file(): return
        with zipfile.ZipFile(archive) as z:
            names=[n.lower() for n in z.namelist()]
            self.assertFalse(any("raw_line" in n or n.endswith(".sas7bdat") or n.endswith(".dbf") for n in names))
            self.assertFalse(any(("manuscript" in Path(n).name and "instructions" not in Path(n).name) or "publication_figure" in n for n in names))
            bad=z.testzip();self.assertIsNone(bad)


if __name__=="__main__":unittest.main()
