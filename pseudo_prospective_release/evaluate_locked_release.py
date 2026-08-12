#!/usr/bin/env python3
"""Score a locked release after outcomes arrive; this script never refits models."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--outcomes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    pred=pd.read_parquet(a.release/'health_region_forecasts.parquet')
    obs=pd.read_parquet(a.outcomes) if a.outcomes.suffix=='.parquet' else pd.read_csv(a.outcomes)
    key=['forecast_origin','health_region_code']
    frame=pred.merge(obs[key+['observed']],on=key,how='inner',validate='one_to_one')
    if len(frame)!=len(pred): raise ValueError('Outcomes do not exactly cover the locked release.')
    y=frame.observed.to_numpy(float); total=.5*np.abs(y-frame.q50.to_numpy(float))
    for alpha,lo,hi in ((.5,25,75),(.2,10,90),(.1,5,95)):
        lower=frame[f'q{lo:02d}'].to_numpy(float); upper=frame[f'q{hi:02d}'].to_numpy(float)
        total+=(alpha/2)*(upper-lower+(2/alpha)*(lower-y)*(y<lower)+(2/alpha)*(y-upper)*(y>upper))
    result={'status':'LOCKED_RELEASE_EVALUATED_WITHOUT_REFIT','rows':len(frame),'mae':float(np.mean(np.abs(y-frame['mean']))),'bias':float(np.mean(frame['mean']-y)),'standard_WIS':float(np.mean(total/3.5))}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
