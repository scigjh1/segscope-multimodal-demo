"""Run a reproducible end-to-end SegScope benchmark."""

from __future__ import annotations

import argparse
import base64
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from server import segment


ROOT = Path(__file__).resolve().parent


def data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["fast", "precise"], default="fast")
    parser.add_argument("--output", default="benchmark_results/local_cpu.json")
    args = parser.parse_args()

    payload = {
        "image": data_url(ROOT / "static" / "assets" / "sample_rgb.png"),
        "auxiliary": data_url(ROOT / "static" / "assets" / "sample_aux.png"),
        "roi": [115, 85, 570, 365],
        "iterations": args.iterations,
        "mode": args.mode,
        "fusionWeight": 0.35,
    }

    for _ in range(args.warmup):
        segment(payload)

    wall_times = []
    engine_times = []
    result = None
    for _ in range(args.runs):
        started = time.perf_counter()
        result = segment(payload)
        wall_times.append((time.perf_counter() - started) * 1000)
        engine_times.append(result["metrics"]["processingMs"])

    assert result is not None
    image = cv2.imread(str(ROOT / "static" / "assets" / "sample_rgb.png"))
    output = {
        "timestampUtc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "device": "CPU",
        },
        "protocol": {
            "inputWidth": int(image.shape[1]),
            "inputHeight": int(image.shape[0]),
            "warmupRuns": args.warmup,
            "measuredRuns": args.runs,
            "grabCutIterations": args.iterations,
            "mode": args.mode,
            "fusionWeight": 0.35,
            "scope": "JSON-ready end-to-end function including decode, segmentation, fusion, metrics and PNG encode",
        },
        "latencyMs": {
            "mean": round(statistics.fmean(wall_times), 2),
            "median": round(statistics.median(wall_times), 2),
            "p95": round(percentile(wall_times, 0.95), 2),
            "min": round(min(wall_times), 2),
            "max": round(max(wall_times), 2),
        },
        "throughputFps": round(1000 / statistics.fmean(wall_times), 2),
        "engineReportedMeanMs": round(statistics.fmean(engine_times), 2),
        "sampleOutput": result["metrics"],
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
