"""SegScope local server and OpenCV segmentation API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import importlib.util
from functools import lru_cache
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MAX_BODY_BYTES = 24 * 1024 * 1024
MAX_IMAGE_EDGE = 512


def decode_image(data_url: str, grayscale: bool = False) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise ValueError("图片数据无效")
    encoded = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("图片编码无效") from exc
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("无法读取图片")
    return image


def resize_for_inference(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_EDGE / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def normalize_roi(raw_roi: list[float] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not raw_roi or len(raw_roi) != 4:
        margin_x = max(1, round(width * 0.08))
        margin_y = max(1, round(height * 0.08))
        return margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y

    x, y, roi_width, roi_height = (round(float(value)) for value in raw_roi)
    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    roi_width = max(2, min(roi_width, width - x))
    roi_height = max(2, min(roi_height, height - y))
    return x, y, roi_width, roi_height


def grabcut_mask(image: np.ndarray, roi: tuple[int, int, int, int], iterations: int) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image,
        mask,
        roi,
        background_model,
        foreground_model,
        max(1, min(iterations, 10)),
        cv2.GC_INIT_WITH_RECT,
    )
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if cv2.countNonZero(binary) < max(12, int(height * width * 0.002)):
        x, y, roi_width, roi_height = roi
        binary[y : y + roi_height, x : x + roi_width] = 255
    return binary


def fast_color_mask(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Segment by comparing each Lab pixel with central-foreground and border-background seeds."""
    height, width = image.shape[:2]
    x, y, roi_width, roi_height = roi
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    inset_x = max(2, round(roi_width * 0.3))
    inset_y = max(2, round(roi_height * 0.3))
    center = lab[
        y + inset_y : y + roi_height - inset_y,
        x + inset_x : x + roi_width - inset_x,
    ]
    if center.size == 0:
        center = lab[y : y + roi_height, x : x + roi_width]

    border = np.concatenate(
        [
            lab[: max(1, y), :].reshape(-1, 3),
            lab[min(height - 1, y + roi_height) :, :].reshape(-1, 3),
            lab[y : y + roi_height, : max(1, x)].reshape(-1, 3),
            lab[y : y + roi_height, min(width - 1, x + roi_width) :].reshape(-1, 3),
        ],
        axis=0,
    )
    if border.size == 0:
        border = lab.reshape(-1, 3)

    foreground_color = np.median(center.reshape(-1, 3), axis=0)
    background_color = np.median(border, axis=0)
    foreground_distance = np.linalg.norm(lab - foreground_color, axis=2)
    background_distance = np.linalg.norm(lab - background_color, axis=2)
    confidence = background_distance - foreground_distance
    binary = np.where(confidence > 0, 255, 0).astype(np.uint8)

    roi_mask = np.zeros((height, width), np.uint8)
    roi_mask[y : y + roi_height, x : x + roi_width] = 255
    binary = cv2.bitwise_and(binary, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def auxiliary_probability(auxiliary: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    auxiliary = cv2.GaussianBlur(auxiliary, (5, 5), 0)
    normalized = cv2.normalize(auxiliary, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    x, y, roi_width, roi_height = roi
    inside = normalized[y : y + roi_height, x : x + roi_width]
    outside_mask = np.ones_like(normalized, dtype=bool)
    outside_mask[y : y + roi_height, x : x + roi_width] = False
    outside = normalized[outside_mask]
    if outside.size and float(inside.mean()) < float(outside.mean()):
        normalized = 1.0 - normalized
    return normalized


def fuse_masks(
    primary_mask: np.ndarray,
    auxiliary: np.ndarray | None,
    roi: tuple[int, int, int, int],
    weight: float,
) -> np.ndarray:
    if auxiliary is None:
        return primary_mask.copy()

    probability = auxiliary_probability(auxiliary, roi)
    primary_probability = primary_mask.astype(np.float32) / 255.0
    weight = float(np.clip(weight, 0.0, 0.85))
    fused_probability = primary_probability * (1.0 - weight) + probability * weight
    fused = np.where(fused_probability >= 0.5, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel, iterations=2)
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=1)
    x, y, roi_width, roi_height = roi
    roi_mask = np.zeros_like(fused)
    roi_mask[y : y + roi_height, x : x + roi_width] = 255
    return cv2.bitwise_and(fused, roi_mask)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :] = (72, 184, 112)
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image[selected], 0.52, tint[selected], 0.48, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (32, 104, 255), 2, cv2.LINE_AA)
    return overlay


def encode_png(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("结果图片编码失败")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def mask_metrics(mask: np.ndarray, primary_mask: np.ndarray, processing_ms: int) -> dict[str, float | int]:
    total_pixels = mask.shape[0] * mask.shape[1]
    selected_pixels = cv2.countNonZero(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    component_count = sum(1 for contour in contours if cv2.contourArea(contour) >= 12)
    boundary_length = sum(cv2.arcLength(contour, True) for contour in contours)
    intersection = cv2.countNonZero(cv2.bitwise_and(mask, primary_mask))
    union = cv2.countNonZero(cv2.bitwise_or(mask, primary_mask))
    agreement = intersection / union if union else 1.0
    return {
        "processingMs": processing_ms,
        "areaRatio": round(selected_pixels / total_pixels * 100, 1),
        "components": component_count,
        "boundaryLength": round(boundary_length, 1),
        "agreement": round(agreement * 100, 1),
    }


def segment(payload: dict) -> dict:
    started = time.perf_counter()
    image, scale = resize_for_inference(decode_image(payload.get("image", "")))
    height, width = image.shape[:2]
    raw_roi = payload.get("roi")
    if raw_roi and scale != 1.0:
        raw_roi = [float(value) * scale for value in raw_roi]
    roi = normalize_roi(raw_roi, width, height)
    iterations = int(payload.get("iterations", 5))
    weight = float(payload.get("fusionWeight", 0.35))
    mode = payload.get("mode", "fast")
    if mode not in {"fast", "precise", "private"}:
        raise ValueError("推理模式无效")

    auxiliary = None
    if payload.get("auxiliary"):
        auxiliary = decode_image(payload["auxiliary"], grayscale=True)
        auxiliary = cv2.resize(auxiliary, (width, height), interpolation=cv2.INTER_AREA)

    if mode == "private":
        primary = private_model_mask(image, auxiliary, roi)
    else:
        primary = fast_color_mask(image, roi) if mode == "fast" else grabcut_mask(image, roi, iterations)
    fused = fuse_masks(primary, auxiliary, roi, weight)
    overlay = make_overlay(image, fused)
    processing_ms = round((time.perf_counter() - started) * 1000)

    return {
        "width": width,
        "height": height,
        "primaryMask": encode_png(primary),
        "fusedMask": encode_png(fused),
        "overlay": encode_png(overlay),
        "metrics": mask_metrics(fused, primary, processing_ms),
        "pipeline": (
            "Private local predictor + public postprocessing"
            if mode == "private"
            else
            "OpenCV Lab fast segmentation + auxiliary-guided fusion"
            if mode == "fast" and auxiliary is not None
            else "OpenCV Lab fast segmentation"
            if mode == "fast"
            else "OpenCV GrabCut + auxiliary-guided fusion"
            if auxiliary is not None
            else "OpenCV GrabCut"
        ),
    }


@lru_cache(maxsize=1)
def load_private_predictor(filename: str):
    path = Path(filename).expanduser().resolve()
    if STATIC_DIR.resolve() in path.parents or not path.is_file():
        raise ValueError("私有模型适配器须位于静态资源目录之外")
    spec = importlib.util.spec_from_file_location("segscope_private_predictor", path)
    if spec is None or spec.loader is None:
        raise ValueError("无法加载私有模型适配器")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = getattr(module, "predict", None)
    if not callable(predictor):
        raise ValueError("私有模型适配器须提供 predict 函数")
    return predictor


def private_model_mask(image: np.ndarray, auxiliary, roi) -> np.ndarray:
    filename = os.environ.get("SEG_SCOPE_PREDICTOR_FILE", "")
    if not filename:
        raise ValueError("请先在服务端配置私有模型适配器")
    try:
        mask = np.asarray(load_private_predictor(filename)(image, auxiliary, roi))
    except Exception as exc:
        raise ValueError("私有模型运行失败，请检查本机适配器配置") from exc
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all():
        raise ValueError("私有模型输出须为与输入同尺寸的有限二维掩膜")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("私有模型输出须为 0/1 或 0/255 二值掩膜")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


class SegScopeHandler(BaseHTTPRequestHandler):
    server_version = "SegScope/1.0"

    def send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            self.send_json({"status": "ok", "engine": f"OpenCV {cv2.__version__}", "privatePredictorConfigured": bool(os.environ.get("SEG_SCOPE_PREDICTOR_FILE"))})
            return
        if path == "/README.md":
            content = (ROOT / "README.md").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/":
            path = "/index.html"
        requested = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() not in requested.parents or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = requested.read_bytes()
        content_type, _ = mimetypes.guess_type(requested.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/segment":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("请求体为空或超过 24 MB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json({"ok": True, "result": segment(payload)})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - API boundary
            self.send_json({"ok": False, "error": f"分割失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SegScope multimodal segmentation demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SegScopeHandler)
    print(f"SegScope running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
