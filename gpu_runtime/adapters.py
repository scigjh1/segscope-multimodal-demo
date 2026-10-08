from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch


class SegmentationModel(ABC):
    """Adapters return B,C,D,H,W logits; input spacing is ordered z,y,x."""

    model_id = "unconfigured"
    version = "unknown"
    fingerprint = "unknown"
    description = "Operator-configured model"
    recommended_roi_size = (32, 96, 96)

    @abstractmethod
    def load(self, device: torch.device):
        """Load trusted local weights, returning an eval-mode torch module."""

    def preprocess(self, image, spacing):
        return image

    def predict(self, model, tensor, spacing):
        return model(tensor)

    def postprocess(self, logits):
        if logits.shape[1] == 1:
            return torch.sigmoid(logits.float())
        # Binary foreground union, explicit for multiclass adapters.
        return 1.0 - torch.softmax(logits.float(), dim=1)[:, :1]


class TorchScriptAdapter(SegmentationModel):
    def __init__(self, filename: str, model_id="MySeg-v1", version="1"):
        self.filename = Path(filename).resolve()
        self.model_id, self.version = model_id, version
        self.fingerprint = hashlib.sha256(self.filename.read_bytes()).hexdigest()
        self.description = "Local TorchScript model"

    def load(self, device):
        return torch.jit.load(str(self.filename), map_location=device).eval()


def load_adapter(filename: str) -> SegmentationModel:
    path = Path(filename).expanduser().resolve()
    static = Path(__file__).resolve().parents[1] / "static"
    if static in path.parents or not path.is_file():
        raise ValueError("Adapter must be a trusted file outside static assets")
    spec = importlib.util.spec_from_file_location("segscope_operator_adapter", path)
    if not spec or not spec.loader:
        raise ValueError("Cannot import model adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter = module.create_adapter()
    if not isinstance(adapter, SegmentationModel):
        raise ValueError("create_adapter() must return SegmentationModel")
    return adapter
