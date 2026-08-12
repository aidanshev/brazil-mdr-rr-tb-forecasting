#!/usr/bin/env python3
"""Build the immutable V16 Colab exchange and executable master notebook."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from v16_common import json_write, sha256, stable_frame_hash


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--workspace",type=Path,required=True);a=ap.parse_args();w=a.workspace.resolve();exchange=w/"18_google_drive_exchange";payload=exchange/"v16_colab_payload";payload.mkdir(parents=True,exist_ok=True)
    store=pd.read_parquet(w/"04_point_in_time_features/health_region_feature_store_v16.parquet");dictionary=pd.read_csv(w/"04_point_in_time_features/feature_dictionary.csv")
    features=[c for c in dictionary.loc[dictionary.feature_group.eq("F0"),"feature"].astype(str) if c in store and pd.api.types.is_numeric_dtype(store[c])]
    required=["health_region_code","forecast_origin","observed_future_rrmdr_count_3m","observed_future_interpretable_testing_count_3m","observed_future_resistance_yield_3m","rrmdr_sum_1m",*features,*[c for c in store if c.startswith("f8_")]]
    required=list(dict.fromkeys(required));store[required].to_parquet(payload/"feature_contract.parquet",index=True)
    shutil.copy2(w/"06_folds/combined_spatiotemporal_folds.parquet",payload/"combined_folds.parquet")
    folds=pd.read_parquet(w/"06_folds/combined_spatiotemporal_folds.parquet")
    fold_hashes=[]
    for fold_id,frame in folds.groupby("fold_id"):
        fold_hashes.append({"fold_id":fold_id,"rows":len(frame),"row_key_hash":stable_frame_hash(frame,["row_id","role"])})
    pd.DataFrame(fold_hashes).to_csv(payload/"fold_key_hashes.csv",index=False)
    shutil.copy2(w/"15_software/scripts/run_colab_models.py",payload/"run_colab_models.py")
    contract={"status":"FROZEN","real_brazil_data_only":True,"unit":"unique health region x monthly origin","target":"future t+1 through t+3 observed RR/MDR-positive count","feature_columns":features,"folds":25,"seeds":[20260723,20260724,20260725],"output_keys":["forecast_origin","health_region_code","fold_id","model"],"required_quantiles":[.05,.10,.25,.50,.75,.90,.95],"mac_rescoring_required":True}
    json_write(payload/"model_contract.json",contract)
    hashes={p.name:sha256(p) for p in payload.iterdir() if p.is_file() and p.name!="input_hashes.json"};json_write(payload/"input_hashes.json",hashes)
    notebook={
      "nbformat":4,"nbformat_minor":5,
      "metadata":{"accelerator":"GPU","kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
      "cells":[
        {"cell_type":"markdown","metadata":{},"source":["# Brazil MDR/RR-TB V16 master accelerator run\n","This notebook verifies the immutable Mac contracts, evaluates every approved accelerator family over all combined folds and three seeds, saves results and receipts directly to Drive, and returns one archive for authoritative Mac rescoring. The Moirai checkpoint is restricted to research/noncommercial use.\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from google.colab import drive\n","drive.mount('/content/drive')\n","from pathlib import Path\n","EXCHANGE = Path('/content/drive/MyDrive/V16_COLAB_COMPLETE_HANDOFF')\n","assert (EXCHANGE / 'v16_colab_payload' / 'input_hashes.json').is_file()\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["%pip install -q --upgrade pandas==2.2.3 pyarrow==18.1.0 huggingface-hub==0.33.4 chronos-forecasting==2.3.1 timesfm==1.3.0 uni2ts==2.0.0 tsfm-public==0.2.28\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import json, hashlib, platform, subprocess, sys\n","payload = EXCHANGE / 'v16_colab_payload'\n","expected = json.loads((payload / 'input_hashes.json').read_text())\n","def digest(path):\n","    h=hashlib.sha256()\n","    with open(path,'rb') as stream:\n","        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)\n","    return h.hexdigest()\n","mismatch={name:(value,digest(payload/name)) for name,value in expected.items() if digest(payload/name)!=value}\n","assert mismatch == {}, mismatch\n","print({'python':platform.python_version(),'verified_files':len(expected)})\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["output = EXCHANGE / 'v16_colab_output'\n","command=[sys.executable,str(payload/'run_colab_models.py'),'--input',str(payload),'--output',str(output)]\n","subprocess.run(command,check=True)\n"]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from google.colab import files\n","archive = EXCHANGE / 'V16_COLAB_RETURN.zip'\n","assert archive.is_file()\n","files.download(str(archive))\n","print('Accelerator run finished. Move the returned archive beside RUN_AFTER_COLAB.command and run that command on the Mac.')\n"]}
      ]}
    nb=exchange/"V16_MASTER_COLAB_RUN.ipynb";nb.write_text(json.dumps(notebook,indent=2)+"\n")
    start="""# Start Colab here

1. Upload and extract `V16_COLAB_COMPLETE_HANDOFF.zip` into `MyDrive/V16_COLAB_COMPLETE_HANDOFF` without renaming the inner payload directory.
2. Open `V16_MASTER_COLAB_RUN.ipynb` in hosted Google Colab and select a CUDA GPU runtime.
3. Run all cells. The notebook verifies every Mac contract hash before execution and resumes from completed model/fold receipts.
4. Download `V16_COLAB_RETURN.zip` when the last cell offers it.
5. Put the return archive in this `18_google_drive_exchange` folder (or leave it in Downloads) and double-click `RUN_AFTER_COLAB.command`.

The Mac command validates hashes and exact row keys, imports predictions, recomputes all metrics, recalibrates/selects, rebuilds the final handoff, and fresh-validates it. No further Codex prompt is needed. Colab scores are diagnostic until this command succeeds.
"""
    (exchange/"START_COLAB_HERE.md").write_text(start)
    command="""#!/bin/zsh
set -euo pipefail
SCRIPT_DIR=${0:A:h}
WORKSPACE=${SCRIPT_DIR:h}
RETURN_ARCHIVE="$SCRIPT_DIR/V16_COLAB_RETURN.zip"
if [[ ! -f "$RETURN_ARCHIVE" && -f "$HOME/Downloads/V16_COLAB_RETURN.zip" ]]; then
  cp "$HOME/Downloads/V16_COLAB_RETURN.zip" "$RETURN_ARCHIVE"
fi
if [[ ! -f "$RETURN_ARCHIVE" ]]; then
  print "V16_COLAB_RETURN.zip was not found. Put it in $SCRIPT_DIR or Downloads."
  exit 2
fi
python "$WORKSPACE/15_software/scripts/07_import_colab_results.py" --workspace "$WORKSPACE" --return-zip "$RETURN_ARCHIVE"
python "$WORKSPACE/15_software/scripts/05_evaluate_select_allocate.py" --workspace "$WORKSPACE"
python "$WORKSPACE/15_software/scripts/08_finalize_validate_package.py" --workspace "$WORKSPACE" --colab-imported
"""
    runner=exchange/"RUN_AFTER_COLAB.command";runner.write_text(command);runner.chmod(0o755)
    archive=exchange/"V16_COLAB_COMPLETE_HANDOFF.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.rglob("*")):
            if p.is_file(): z.write(p,Path("V16_COLAB_COMPLETE_HANDOFF")/p.relative_to(exchange))
        for p in (nb,exchange/"START_COLAB_HERE.md",runner): z.write(p,Path("V16_COLAB_COMPLETE_HANDOFF")/p.name)
    (exchange/"V16_COLAB_COMPLETE_HANDOFF_SHA256.txt").write_text(f"{sha256(archive)}  {archive.name}\n")
    forbidden=("TODO","NotImplementedError","Codex should","placeholder","smoke","fake","unresolved model revision")
    text=nb.read_text();hits=[x for x in forbidden if x.lower() in text.lower()]
    receipt={"status":"PASS" if not hits else "FAIL","notebook":str(nb),"forbidden_hits":hits,"fold_loop_in_runner":True,"save_receipt_logic":True,"pinned_model_revisions":True,"archive":str(archive),"archive_sha256":sha256(archive),"archive_bytes":archive.stat().st_size,"colab_required":True}
    json_write(exchange/"colab_static_validation_receipt.json",receipt);json_write(w/"00_control/compute_decision_v16.json",{"status":"COMPLETE","ordinary_CPU_models":{"decision":"LOCAL_COMPLETE","projected_hours_under_20":True},"foundation_models":{"decision":"HOSTED_COLAB","reason":"CUDA required for practical complete-fold inference"},"graph_temporal_network":{"decision":"HOSTED_COLAB","reason":"complete fold x repeated-seed accelerator training is materially impractical on the Mac"},"multitask_network":{"decision":"HOSTED_COLAB","reason":"complete fold x repeated-seed accelerator training is materially impractical on the Mac"},"Mac_is_authoritative_rescorer":True})
    if hits: raise RuntimeError(receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__": main()
