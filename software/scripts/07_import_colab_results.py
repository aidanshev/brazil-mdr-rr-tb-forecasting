#!/usr/bin/env python3
"""Validate a Colab return archive, import OOF predictions, and record Mac acceptance."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from v16_common import json_write, sha256, stable_frame_hash

MODELS=("chronos_2","timesfm_2","moirai_2_small","tiny_time_mixer_r2","graph_temporal_network","multitask_burden_testing_yield")
KEYS=["forecast_origin","health_region_code","fold_id"]


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--workspace",type=Path,required=True);ap.add_argument("--return-zip",type=Path,required=True);a=ap.parse_args();w=a.workspace.resolve();archive=a.return_zip.resolve()
    if not archive.is_file(): raise FileNotFoundError(archive)
    with tempfile.TemporaryDirectory(dir=w/"21_tmp") as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                destination=(root/member.filename).resolve()
                if root.resolve() not in destination.parents and destination!=root.resolve(): raise RuntimeError("Unsafe archive member")
            z.extractall(root)
        output=root/"v16_colab_output";manifest=json.loads((output/"output_hashes.json").read_text());bad={name:{"expected":digest,"actual":sha256(output/name)} for name,digest in manifest.items() if not (output/name).is_file() or sha256(output/name)!=digest}
        if bad: raise RuntimeError(f"Returned output hash mismatch: {bad}")
        statuses={};completed=[]
        for model in MODELS:
            done=output/f"{model}_completion_receipt.json";failed=output/f"{model}_failure_receipt.json"
            if done.is_file() == failed.is_file(): raise RuntimeError(f"{model} must have exactly one completion or failure receipt")
            statuses[model]="COMPLETE" if done.is_file() else "EXPLICIT_FAILURE"
            if done.is_file(): completed.append(model)
        if completed:
            oof=pd.read_parquet(output/"all_colab_oof.parquet");oof.forecast_origin=pd.to_datetime(oof.forecast_origin);oof.health_region_code=oof.health_region_code.astype(str)
            expected=pd.read_parquet(w/"09_oof_predictions/combined_spatiotemporal_oof_predictions.parquet");expected=expected[expected.model.eq("simple_median_ensemble")][KEYS].sort_values(KEYS).reset_index(drop=True)
            for model,frame in oof.groupby("model"):
                actual=frame[KEYS].sort_values(KEYS).reset_index(drop=True)
                if not actual.equals(expected): raise RuntimeError(f"OOF key mismatch for {model}")
            context=pd.read_parquet(w/"09_oof_predictions/combined_spatiotemporal_oof_predictions.parquet");context=context[context.model.eq("simple_median_ensemble")][KEYS+["positive_observed_excess","negative_binomial_counterfactual_mean","rrmdr_sum_3m"]]
            oof=oof.merge(context,on=KEYS,how="left",validate="many_to_one");oof["predicted_excess"]=oof["mean"]-oof["negative_binomial_counterfactual_mean"];oof.to_parquet(w/"09_oof_predictions/v16_colab_model_oof.parquet",index=False)
        destination=w/"18_google_drive_exchange"/f"imported_colab_output_{sha256(archive)[:12]}"
        shutil.copytree(output,destination)
    receipt={"status":"PASS","archive":str(archive),"archive_sha256":sha256(archive),"models_completed":len(completed),"models_failed":len(MODELS)-len(completed),"model_statuses":statuses,"hashes_verified":True,"exact_oof_keys_verified":True if completed else "NO_COMPLETED_OOF","Mac_metrics_recomputed":False}
    json_write(w/"18_google_drive_exchange/colab_import_receipt.json",receipt);print(json.dumps(receipt,indent=2))


if __name__=="__main__": main()
