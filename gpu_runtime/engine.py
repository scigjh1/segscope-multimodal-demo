from __future__ import annotations

from contextlib import nullcontext
import threading
import time
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .adapters import SegmentationModel


def device_inventory():
    devices = [{"id": "cpu", "name": "CPU", "precision": ["fp32"]}]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            devices.append({"id": f"cuda:{i}", "name": props.name,
                            "total_mib": round(total / 2**20), "free_mib": round(free / 2**20),
                            "precision": ["fp32", "fp16"]})
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "devices": devices}


class InferenceEngine:
    def __init__(self, adapter: SegmentationModel):
        self.adapter = adapter
        self.models = {}
        self.lock = threading.RLock()

    def metadata(self):
        return {"id": self.adapter.model_id, "version": self.adapter.version,
                "fingerprint": self.adapter.fingerprint, "description": self.adapter.description,
                "recommended_roi_size": list(self.adapter.recommended_roi_size)}

    def model(self, device):
        key = str(device)
        if key not in self.models:
            self.models[key] = self.adapter.load(device).to(device).eval()
        return self.models[key]

    def infer(self, volume, spacing=(1, 1, 1), *, device="cpu", precision="fp32",
              roi_size=(32, 96, 96), batch_size=1, overlap=0.25, resample_spacing=None):
        if precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        target_device = torch.device(device)
        if target_device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA are supported")
        if target_device.type == "cuda":
            if not torch.cuda.is_available() or (target_device.index or 0) >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device is unavailable; no silent CPU fallback")
        elif precision != "fp32":
            raise ValueError("FP16 mode requires CUDA")
        array = np.asarray(volume, dtype=np.float32)
        if array.ndim not in (3, 4) or not np.isfinite(array).all() or array.size > 128**3 * 128:
            raise ValueError("Expected finite D,H,W or B,D,H,W volume")
        if min(array.shape) < 1 or any(int(v) < 16 or int(v) % 16 for v in roi_size):
            raise ValueError("ROI dimensions must be positive multiples of 16")
        if not 0 <= overlap < 1 or not 1 <= batch_size <= 8:
            raise ValueError("Invalid overlap or sliding-window batch size")
        spacing = tuple(float(x) for x in spacing)
        if len(spacing) != 3 or min(spacing) <= 0 or not np.isfinite(spacing).all():
            raise ValueError("Spacing must contain three finite positive values")
        with self.lock, torch.inference_mode():
            model = self.model(target_device)  # Cold load excluded from steady-state timing.
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
                torch.cuda.reset_peak_memory_stats(target_device)
            started = time.perf_counter()
            original_shape = array.shape[-3:]
            values = self.adapter.preprocess(array.copy(), spacing)
            host = torch.from_numpy(np.ascontiguousarray(values)).float()
            host = host[None, None] if host.ndim == 3 else host[:, None]
            effective_spacing = spacing
            if resample_spacing is not None:
                new_spacing = tuple(float(x) for x in resample_spacing)
                if len(new_spacing) != 3 or min(new_spacing) <= 0 or not np.isfinite(new_spacing).all():
                    raise ValueError("Invalid resample spacing")
                shape = tuple(max(1, round(n * s / t)) for n, s, t in zip(original_shape, spacing, new_spacing))
                if np.prod(shape) > 128**3 * 128:
                    raise ValueError("Resampled volume is too large")
                host = torch.nn.functional.interpolate(host, size=shape, mode="trilinear", align_corners=False)
                effective_spacing = new_spacing
            pre_end = time.perf_counter()
            data = host.to(target_device)
            spacing_tensor = torch.tensor(effective_spacing, dtype=torch.float32, device=target_device)[None]
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            transfer_end = time.perf_counter()

            def predict(tensor):
                logits = self.adapter.predict(model, tensor, spacing_tensor.expand(tensor.shape[0], -1))
                if not isinstance(logits, torch.Tensor) or logits.ndim != 5:
                    raise ValueError("Adapter must return B,C,D,H,W logits")
                return logits.float()

            autocast = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
            with autocast:
                logits = sliding_window_inference(data, tuple(roi_size), batch_size, predict,
                                                  overlap=overlap, mode="gaussian", progress=False)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            inference_end = time.perf_counter()
            probability = self.adapter.postprocess(logits)
            if tuple(probability.shape[-3:]) != original_shape:
                probability = torch.nn.functional.interpolate(probability, size=original_shape,
                                                               mode="trilinear", align_corners=False)
            probability = probability[:, 0].float().cpu().numpy()
            if not np.isfinite(probability).all():
                raise RuntimeError("Non-finite model prediction")
            mask = (probability >= 0.5).astype(np.uint8)
            clipped = np.clip(probability, 1e-6, 1 - 1e-6)
            entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            ended = time.perf_counter()
            timings = {"preprocess_ms": (pre_end-started)*1000, "h2d_ms": (transfer_end-pre_end)*1000,
                       "inference_ms": (inference_end-transfer_end)*1000,
                       "postprocess_d2h_ms": (ended-inference_end)*1000, "total_ms": (ended-started)*1000}
            timing_method = "synchronized perf_counter wall time; preprocessing + H2D + MONAI windows + postprocess/D2H"
            memory = {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
            if target_device.type == "cuda":
                memory = {"peak_allocated_mib": torch.cuda.max_memory_allocated(target_device) / 2**20,
                          "peak_reserved_mib": torch.cuda.max_memory_reserved(target_device) / 2**20}
            return {"mask": mask[0] if array.ndim == 3 else mask,
                    "probability": probability[0] if array.ndim == 3 else probability,
                    "uncertainty": entropy[0] if array.ndim == 3 else entropy,
                    "timings": timings, "timing_method": timing_method, "memory": memory,
                    "device": str(target_device), "precision": precision, "model": self.metadata(),
                    "input_shape": list(array.shape), "native_shape": list(original_shape),
                    "roi_size": list(roi_size), "window_batch_size": batch_size, "overlap": overlap}
