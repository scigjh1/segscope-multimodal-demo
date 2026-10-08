"""Reproducible CPU/CUDA/AMP timings. Never invent clinical accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import torch
import monai

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gpu_runtime.adapters import load_adapter
from gpu_runtime.engine import InferenceEngine,device_inventory
from gpu_runtime.metrics import evaluate,regression


def summary(rows):
    latency=np.array([r['timings']['total_ms'] for r in rows])
    return {'p50_ms':float(np.percentile(latency,50)),'p95_ms':float(np.percentile(latency,95)),
            'mean_ms':float(latency.mean()),'throughput_volumes_s':1000*rows[0]['input_shape'][0]/float(latency.mean()),
            'peak_allocated_mib':max(x['memory']['peak_allocated_mib'] for x in rows),
            'peak_reserved_mib':max(x['memory']['peak_reserved_mib'] for x in rows),
            'stages_mean_ms':{k:float(np.mean([x['timings'][k] for x in rows])) for k in rows[0]['timings']}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--gpu',default='cuda:1')
    parser.add_argument('--shape',nargs=3,type=int,default=[32,96,96])
    parser.add_argument('--roi',nargs=3,type=int,default=[32,96,96])
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--runs',type=int,default=30)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--window-batch',type=int,default=1)
    parser.add_argument('--cpu-threads',type=int,default=8)
    parser.add_argument('--gpu-only',action='store_true')
    args=parser.parse_args()
    if args.warmup<1 or args.runs<5 or not 1<=args.batch<=8:
        parser.error('Use warmup>=1, runs>=5, batch 1..8')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Fixed synthetic CT-like tensor; measures runtime, not diagnostic quality.
    rng=np.random.default_rng(42)
    data=rng.normal(40,80,(args.batch,*args.shape)).astype(np.float32)
    engine=InferenceEngine(load_adapter(args.adapter))
    report={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'model':engine.metadata(),'hardware':device_inventory(),'monai':monai.__version__,
            'python':platform.python_version(),'cpu':platform.processor(),'cpu_threads':args.cpu_threads,
            'workload':{'kind':'fixed synthetic 3D CT-like tensor; performance only','seed':42,
                        'shape_bdhw':list(data.shape),'spacing_zyx_mm':[2.5,1.2,1.2],
                        'input_sha256':hashlib.sha256(data.tobytes()).hexdigest(),
                        'roi_size':args.roi,'window_batch_size':args.window_batch,'overlap':0.25},
            'protocol':{'warmup':args.warmup,'measured':args.runs,'cold_model_load_excluded':True,
                        'disk_network_excluded':True,'tf32':False,'cudnn_benchmark':False,
                        'latency':'synchronized end-to-end runtime wall time; each sample is one complete batch',
                        'vram':'PyTorch allocated/reserved process high-water mark; not total nvidia-smi memory',
                        'sampling':'same seeded input repeated; quantiles use numpy linear interpolation'},
            'results':[],'clinical_accuracy':None}
    try:
        report['contention']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                                                      '--format=csv,noheader'],text=True).strip().splitlines()
    except Exception:report['contention']=[]
    outputs={}
    modes=[] if args.gpu_only else [('cpu','fp32')]
    modes += [(args.gpu,'fp32'),(args.gpu,'fp16')]
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    for device,precision in modes:
        rows=[]
        print('RUN',device,precision,flush=True)
        for i in range(args.warmup+args.runs):
            result=engine.infer(data,(2.5,1.2,1.2),device=device,precision=precision,
                                roi_size=args.roi,batch_size=args.window_batch)
            if i>=args.warmup:
                rows.append({k:result[k] for k in ('timings','memory','input_shape')})
            if i==args.warmup+args.runs-1:
                outputs[device+'/'+precision]=result['probability']
        entry={'device':device,'precision':precision,'summary':summary(rows),'raw_samples':rows}
        report['results'].append(entry)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'device':device,'precision':precision,**entry['summary']}),flush=True)
    fp32=outputs[args.gpu+'/fp32'];fp16=outputs[args.gpu+'/fp16']
    report['precision_regression']={'kind':'same checkpoint FP32 vs AMP FP16, not two trained model versions',
                                   'max_abs_probability_error':float(np.max(np.abs(fp32-fp16))),
                                   'mean_abs_probability_error':float(np.mean(np.abs(fp32-fp16))),
                                   'mask_agreement':float(np.mean((fp32>=0.5)==(fp16>=0.5))),
                                   'mask_dice':evaluate(fp16[0]>=0.5,fp32[0]>=0.5)['dice'],
                                   'fp32_foreground_voxels':int((fp32>=.5).sum()),
                                   'fp16_foreground_voxels':int((fp16>=.5).sum()),
                                   'empty_prediction_comparison':not bool((fp32>=.5).any() or (fp16>=.5).any()),
                                   'quality_note':'Synthetic runtime input; empty-mask agreement is not evidence of segmentation accuracy',
                                   'performance':regression(report['results'][-2]['summary'],report['results'][-1]['summary'])}
    if not args.gpu_only:
        cpu=report['results'][0]['summary']['p50_ms']
        report['speedup_vs_cpu_p50']={r['precision']:cpu/r['summary']['p50_ms'] for r in report['results'][1:]}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('SAVED',str(out),flush=True)


if __name__=='__main__':main()
