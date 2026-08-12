#!/usr/bin/env python3
"""GPU runner for the frozen V16 contracts; intended for hosted Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260723
QUANTILES = (.05,.10,.25,.50,.75,.90,.95)
MODELS = {
    "chronos_2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", "apache-2.0"),
    "timesfm_2": ("google/timesfm-2.0-500m-pytorch", "dc2443792ce5516872b89b37cf1bc058c3bf0c10", "apache-2.0"),
    "moirai_2_small": ("Salesforce/moirai-2.0-R-small", "30f43ff08c8494f4943ae1521e9d4e94a0fbb389", "cc-by-nc-4.0 research/noncommercial"),
    "tiny_time_mixer_r2": ("ibm-granite/granite-timeseries-ttm-r2", "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f", "apache-2.0"),
}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+"\n")


def verify_inputs(root: Path) -> None:
    manifest=json.loads((root/"input_hashes.json").read_text())
    bad={name:{"expected":digest,"actual":sha256(root/name)} for name,digest in manifest.items() if not (root/name).is_file() or sha256(root/name)!=digest}
    if bad: raise RuntimeError(f"Frozen input hash mismatch: {bad}")


def residual_quantiles(y: np.ndarray, fitted: np.ndarray, pred: np.ndarray) -> dict[str,np.ndarray]:
    residual=np.asarray(y,float)-np.asarray(fitted,float);stack=np.sort(np.vstack([np.maximum(pred+np.nanquantile(residual,q),0) for q in QUANTILES]),axis=0)
    return {f"q{int(q*100):02d}":stack[i] for i,q in enumerate(QUANTILES)}


def torch_family(name: str, x: np.ndarray, target: np.ndarray, aux: np.ndarray, edge: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int, checkpoint: Path) -> tuple[np.ndarray,np.ndarray]:
    import torch
    torch.manual_seed(seed);np.random.seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    med=np.nanmedian(x[train],axis=0);scale=np.nanpercentile(x[train],75,axis=0)-np.nanpercentile(x[train],25,axis=0);scale=np.where(scale>1e-8,scale,1)
    z=np.nan_to_num((x-med)/scale).astype("float32"); graph=np.nan_to_num(edge).astype("float32")
    if name=="graph_temporal_network": design=np.concatenate([z,graph],axis=1); outputs=1
    else: design=z; outputs=aux.shape[1]
    net=torch.nn.Sequential(torch.nn.Linear(design.shape[1],128),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,outputs)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);loss_fn=torch.nn.SmoothL1Loss();xt=torch.from_numpy(design[train]).to(device);yt=torch.from_numpy((target[train,None] if outputs==1 else aux[train]).astype("float32")).to(device)
    best=float("inf");stale=0
    for epoch in range(250):
        net.train();opt.zero_grad();raw=net(xt);loss=loss_fn(torch.nn.functional.softplus(raw),yt);loss.backward();opt.step();value=float(loss.detach().cpu())
        if value<best-1e-6: best=value;stale=0;torch.save({"state_dict":net.state_dict(),"median":med,"scale":scale,"epoch":epoch,"seed":seed},checkpoint)
        else: stale+=1
        if stale>=25: break
    saved=torch.load(checkpoint,map_location=device);net.load_state_dict(saved["state_dict"]);net.eval()
    with torch.no_grad():
        pred=torch.nn.functional.softplus(net(torch.from_numpy(design[test]).to(device))).cpu().numpy()[:,0]
        fitted=torch.nn.functional.softplus(net(torch.from_numpy(design[train]).to(device))).cpu().numpy()[:,0]
    return pred,fitted


def chronos_predict(series: dict[str,np.ndarray], requests: pd.DataFrame, local_model: Path) -> np.ndarray:
    import torch
    from chronos import Chronos2Pipeline
    pipeline=Chronos2Pipeline.from_pretrained(str(local_model),device_map="cuda",torch_dtype=torch.bfloat16)
    frames=[]
    for request_id,r in enumerate(requests.itertuples()):
        history=series[r.health_region_code][:r.position+1]
        frames.append(pd.DataFrame({"id":str(request_id),"timestamp":pd.date_range("2000-01-01",periods=len(history),freq="MS"),"target":history}))
    forecast=pipeline.predict_df(pd.concat(frames,ignore_index=True),prediction_length=3,quantile_levels=[.05,.10,.25,.50,.75,.90,.95],id_column="id",timestamp_column="timestamp",target="target")
    return np.maximum(forecast.groupby("id",sort=False)["predictions"].sum().to_numpy(float),0)


def foundation_family(name: str, store: pd.DataFrame, indexes: np.ndarray, local_model: Path) -> np.ndarray:
    # Each request uses only measurements available at its origin.
    ordered=store.sort_values(["health_region_code","forecast_origin"]);series={k:g.rrmdr_sum_1m.fillna(0).to_numpy(float) for k,g in ordered.groupby("health_region_code")};positions={}
    for _,g in ordered.groupby("health_region_code"):
        for pos,idx in enumerate(g.index): positions[int(idx)]=pos
    req=store.loc[indexes,["health_region_code"]].copy();req["position"]=[positions[int(i)] for i in indexes]
    if name=="chronos_2": return chronos_predict(series,req,local_model)
    if name=="timesfm_2":
        import timesfm
        model=timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend="gpu",per_core_batch_size=32,horizon_len=3),checkpoint=timesfm.TimesFmCheckpoint(path=str(local_model)))
        contexts=[series[r.health_region_code][:r.position+1] for r in req.itertuples()];point,_=model.forecast(contexts,freq=[0]*len(contexts));return np.maximum(np.asarray(point,float).sum(axis=1),0)
    if name=="moirai_2_small":
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        import torch
        module=MoiraiModule.from_pretrained(str(local_model));model=MoiraiForecast(module=module,prediction_length=3,context_length=120,patch_size="auto",num_samples=64,target_dim=1,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
        values=[]
        for r in req.itertuples():
            context=torch.tensor(series[r.health_region_code][:r.position+1][-120:],dtype=torch.float32).reshape(-1,1);samples=model(context).detach().cpu().numpy();values.append(np.maximum(samples.sum(axis=-2).mean(),0))
        return np.asarray(values,float)
    if name=="tiny_time_mixer_r2":
        from tsfm_public import TimeSeriesForecastingPipeline
        pipe=TimeSeriesForecastingPipeline.from_pretrained(str(local_model),device="cuda")
        values=[]
        for r in req.itertuples(): values.append(np.maximum(np.asarray(pipe(series[r.health_region_code][:r.position+1][-512:],prediction_length=3),float).sum(),0))
        return np.asarray(values,float)
    raise ValueError(name)


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();root=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);verify_inputs(root)
    store=pd.read_parquet(root/"feature_contract.parquet");store.forecast_origin=pd.to_datetime(store.forecast_origin);store.health_region_code=store.health_region_code.astype(str);folds=pd.read_parquet(root/"combined_folds.parquet")
    features=json.loads((root/"model_contract.json").read_text())["feature_columns"];x=store[features].replace([np.inf,-np.inf],np.nan).to_numpy(float);target=store.observed_future_rrmdr_count_3m.to_numpy(float);aux=store[["observed_future_rrmdr_count_3m","observed_future_interpretable_testing_count_3m","observed_future_resistance_yield_3m"]].fillna(0).to_numpy(float);edge=store[[c for c in store if c.startswith("f8_")]].fillna(0).to_numpy(float)
    environment={"status":"RECORDED","python":platform.python_version(),"platform":platform.platform(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","UNSET")}
    try:
        import torch
        environment.update({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"})
    except Exception as exc: environment["torch_error"]=f"{type(exc).__name__}: {exc}"
    write_json(out/"environment_receipt.json",environment)
    from huggingface_hub import snapshot_download
    local_models={}
    for name,(repo,revision,license_name) in MODELS.items():
        try:
            local_models[name]=Path(snapshot_download(repo_id=repo,revision=revision,cache_dir=str(root/"hf_cache")))
            write_json(out/f"{name}_license_revision_receipt.json",{"status":"VERIFIED","repository":repo,"revision":revision,"license":license_name})
        except Exception as exc:
            write_json(out/f"{name}_acquisition_failure.json",{"status":"EXPLICIT_FAILURE","repository":repo,"revision":revision,"error":f"{type(exc).__name__}: {exc}"})
    all_outputs=[];seeds=(20260723,20260724,20260725)
    for model_name in [*MODELS,"graph_temporal_network","multitask_burden_testing_yield"]:
        model_outputs=[];started=time.perf_counter();failure=None
        try:
            if model_name in MODELS and model_name not in local_models: raise RuntimeError("Pinned model acquisition did not complete")
            for fold_id in sorted(folds.fold_id.unique()):
                role=folds[folds.fold_id.eq(fold_id)].set_index("row_id").role;train=role.index[role.eq("train")].to_numpy(int);test=role.index[role.eq("test")].to_numpy(int);train=train[np.isfinite(target[train])];test=test[np.isfinite(target[test])]
                if len(test)==0: continue
                seed_predictions=[]
                for seed in seeds:
                    checkpoint=out/"checkpoints"/model_name/f"{fold_id}_seed_{seed}.pt";checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    if model_name in ("graph_temporal_network","multitask_burden_testing_yield"): pred,fitted=torch_family(model_name,x,target,aux,edge,train,test,seed,checkpoint)
                    else:
                        pred=foundation_family(model_name,store,test,local_models[model_name]);fit_sample=train[-min(len(train),439*12):];fitted_sample=foundation_family(model_name,store,fit_sample,local_models[model_name]);fitted=np.interp(np.arange(len(train)),np.linspace(0,len(train)-1,len(fit_sample)),fitted_sample)
                    seed_predictions.append(pred)
                pred=np.median(np.vstack(seed_predictions),axis=0);fit_pred=fitted;qs=residual_quantiles(target[train],fit_pred,pred)
                frame=store.loc[test,["forecast_origin","health_region_code"]].copy();frame["fold_id"]=fold_id;frame["model"]=model_name;frame["observed"]=target[test];frame["mean"]=pred
                for col,val in qs.items():frame[col]=val
                model_outputs.append(frame)
                write_json(out/"fold_receipts"/model_name/f"{fold_id}.json",{"status":"COMPLETE","model":model_name,"fold_id":fold_id,"seeds":list(seeds),"train_rows":len(train),"test_rows":len(test),"real_data":True})
            if not model_outputs: raise RuntimeError("No nonempty evaluation folds")
            joined=pd.concat(model_outputs,ignore_index=True);joined.to_parquet(out/f"{model_name}_oof.parquet",index=False);all_outputs.append(joined)
            write_json(out/f"{model_name}_completion_receipt.json",{"status":"COMPLETE","rows":len(joined),"folds":joined.fold_id.nunique(),"seeds":list(seeds),"wall_seconds":time.perf_counter()-started})
        except Exception as exc:
            failure={"status":"EXPLICIT_FAILURE","model":model_name,"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started}
            write_json(out/f"{model_name}_failure_receipt.json",failure)
    if all_outputs: pd.concat(all_outputs,ignore_index=True).to_parquet(out/"all_colab_oof.parquet",index=False)
    output_files=[p for p in out.rglob("*") if p.is_file()];manifest={str(p.relative_to(out)):sha256(p) for p in output_files};write_json(out/"output_hashes.json",manifest)
    archive=out.parent/"V16_COLAB_RETURN.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(out.parent))
    (out.parent/"V16_COLAB_RETURN_SHA256.txt").write_text(f"{sha256(archive)}  {archive.name}\n")
    print(json.dumps({"status":"COLAB_RUN_FINISHED","archive":str(archive),"models_completed":len(all_outputs),"models_failed":6-len(all_outputs)},indent=2))


if __name__=="__main__": main()
