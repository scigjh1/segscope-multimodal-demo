from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters import load_adapter
from .engine import InferenceEngine, device_inventory
from .io import load_volume, save_volume, phantom
from .metrics import evaluate, failure_mining
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
store = Store(os.environ.get('SEG_SCOPE_DATA_DIR', str(ROOT/'data/gpu_workflow')))
adapter_path = os.environ.get('SEG_SCOPE_MODEL_ADAPTER','')
engine = InferenceEngine(load_adapter(adapter_path)) if adapter_path else None
app = FastAPI(title='SegScope AI GPU Runtime',version='2.0.0')


@app.exception_handler(KeyError)
async def unknown_record(request: Request, exc: KeyError):
    return JSONResponse({'detail':'Case or run not found'},status_code=404)


@app.exception_handler(ValueError)
async def invalid_input(request: Request, exc: ValueError):
    return JSONResponse({'detail':str(exc)},status_code=400)


@app.get('/')
def workbench():
    return FileResponse(ROOT/'static/gpu.html')


@app.middleware('http')
async def origin_boundary(request: Request, call_next):
    origin=request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'Cross-origin access is disabled'},status_code=403)
    return await call_next(request)


def metadata(volume, reference=False):
    return {'shape_zyx':list(volume.array.shape),'spacing_zyx':list(volume.spacing),
            'units':volume.units,'source':volume.source,'reference_available':reference,
            'patient':'Anonymous research case','study':'Segmentation review','series':volume.source}


@app.get('/api/runtime/health')
def health():
    return {'status':'ok',**device_inventory(),'models':[engine.metadata()] if engine else [],
            'adapter_configured':engine is not None,'workload':'Research demo; operator-configured model'}


@app.get('/api/cases')
def cases():
    return store.cases()


@app.post('/api/cases/phantom')
def create_phantom():
    volume, target=phantom()
    meta,directory=store.new_case(metadata(volume,True))
    save_volume(directory/'image.nii.gz',volume.array,volume.affine)
    save_volume(directory/'reference.nii.gz',target,volume.affine)
    return meta


@app.post('/api/cases/upload')
async def upload_case(file: UploadFile=File(...)):
    suffix = next((x for x in ('.nii.gz','.nii','.png','.jpg','.jpeg','.dcm','.dicom','.zip')
                   if (file.filename or '').lower().endswith(x)),None)
    if not suffix:
        raise HTTPException(400,'Unsupported volume format')
    raw=await file.read(128*1024**2+1)
    if len(raw)>128*1024**2:
        raise HTTPException(413,'File exceeds 128 MiB')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/('input'+suffix);path.write_bytes(raw)
        try:
            volume=load_volume(path)
            meta,directory=store.new_case(metadata(volume))
            save_volume(directory/'image.nii.gz',volume.array,volume.affine,volume.units)
            return meta
        except (ValueError,AttributeError,KeyError) as exc:
            raise HTTPException(400,str(exc)) from exc


@app.get('/api/cases/{case_id}/image.nii.gz')
def case_image(case_id: str):
    _,directory=store.case(case_id)
    return FileResponse(directory/'image.nii.gz')


class InferRequest(BaseModel):
    case_id: str
    device: str='cpu'
    precision: str='fp32'
    roi_size: list[int]=Field(default_factory=lambda:[32,96,96])
    window_batch_size: int=Field(default=1,ge=1,le=8)
    overlap: float=Field(default=0.25,ge=0,lt=1)
    resample_spacing: Optional[list[float]]=None


@app.post('/api/runtime/infer')
def infer(request: InferRequest):
    if not engine:
        raise HTTPException(409,'Configure SEG_SCOPE_MODEL_ADAPTER on the server first')
    try:
        meta,directory=store.case(request.case_id)
        volume=load_volume(directory/'image.nii.gz')
        result=engine.infer(volume.array,volume.spacing,device=request.device,precision=request.precision,
                            roi_size=request.roi_size,batch_size=request.window_batch_size,
                            overlap=request.overlap,resample_spacing=request.resample_spacing)
        reference=load_volume(directory/'reference.nii.gz').array if meta['reference_available'] else None
        stats=evaluate(result['mask'],reference,volume.spacing) if reference is not None else None
        mining=failure_mining(result['mask'],result['uncertainty'],reference)
        measured={key:value for key,value in result.items() if key not in ('mask','probability','uncertainty')}
        measured.update(metrics=stats,mining=mining,units=volume.units,
                        foreground_voxels=int(result['mask'].sum()),
                        volume_ml=float(result['mask'].sum()*np.prod(volume.spacing)/1000) if volume.units=='mm' else None,
                        quality_scope=meta.get('reference_scope','Generated phantom reference') if reference is not None else 'No reference mask')
        run,output=store.new_run(request.case_id,measured)
        for name in ('mask','probability','uncertainty'):
            save_volume(output/(name+'.nii.gz'),result[name],volume.affine,volume.units)
        (output/'result.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
        return run
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(400,str(exc)) from exc


class BatchRequest(BaseModel):
    requests: list[InferRequest]=Field(min_length=1,max_length=8)


@app.post('/api/runtime/batch')
def batch(request: BatchRequest):
    # Bounded sequential case jobs; each job can batch sliding windows on GPU.
    rows=[]
    for job in request.requests:
        try:rows.append({'ok':True,'run':infer(job)})
        except HTTPException as exc:rows.append({'ok':False,'error':exc.detail,'case_id':job.case_id})
    return {'jobs':rows,'successes':sum(x['ok'] for x in rows),'total':len(rows)}


@app.get('/api/runs')
def runs():
    return store.runs()


@app.get('/api/runs/{run_id}/{asset}')
def run_asset(run_id: str,asset: str):
    if asset not in ('mask.nii.gz','probability.nii.gz','uncertainty.nii.gz','correction.nii.gz','result.json'):
        raise HTTPException(404,'Unknown asset')
    _,directory=store.run(run_id)
    if not (directory/asset).is_file():
        raise HTTPException(404,'Asset not available')
    return FileResponse(directory/asset)


class ReviewRequest(BaseModel):
    decision: str
    taxonomy: Optional[str]=None
    note: str=''


@app.post('/api/review/{run_id}')
def review(run_id: str,request: ReviewRequest):
    try:
        if request.decision=='edit' and not (store.run(run_id)[1]/'correction.nii.gz').exists():
            raise ValueError('Upload the edited native-space mask before recording edit')
        return store.review(run_id,request.decision,request.note,request.taxonomy)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc


@app.post('/api/review/{run_id}/correction')
async def correction(run_id: str,file: UploadFile=File(...)):
    run,directory=store.run(run_id)
    _,case_directory=store.case(run['case_id'])
    volume=load_volume(case_directory/'image.nii.gz')
    raw=await file.read(128*1024**2+1)
    if len(raw)>128*1024**2:
        raise HTTPException(413,'File exceeds bounds')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'correction.nii.gz';path.write_bytes(raw)
        edited=load_volume(path)
        if edited.array.shape!=volume.array.shape or not np.allclose(edited.affine,volume.affine,atol=1e-3):
            raise HTTPException(400,'Correction must match native image shape and affine')
        if not np.isin(edited.array,[0,1]).all():
            raise HTTPException(400,'Correction must be a binary 0/1 mask')
        save_volume(directory/'correction.nii.gz',edited.array.astype(np.uint8),volume.affine,volume.units)
    return {'ok':True}


@app.get('/api/error-pool')
def error_pool():
    return store.hard_cases()


@app.get('/api/benchmarks')
def benchmarks():
    directory=ROOT/'benchmark_results/gpu'
    return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(directory.glob('*.json'))]


app.mount('/',StaticFiles(directory=ROOT/'static',html=True),name='workbench')
