"""
Resource profiling for detection models.
Measures latency, FPS, memory, FLOPs, and parameter counts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from src.utils.general import ensure_dir, get_device

logger = logging.getLogger(__name__)


class ResourceProfiler:
    """Profile compute resources for object detection models."""

    def __init__(
        self,
        device: str = "auto",
        num_warmup: int = 10,
        num_iterations: int = 100,
        input_size: tuple[int, int, int, int] = (1, 3, 640, 640),
    ):
        self.device = get_device(device)
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.input_size = input_size
        self.results: list[dict] = []

    def _count_parameters(self, model: torch.nn.Module) -> dict[str, float]:
        """Count model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "total_params_m": total / 1e6,
        }

    def _measure_flops(self, model: torch.nn.Module) -> float | None:
        """Measure FLOPs using fvcore or thop."""
        dummy = torch.randn(*self.input_size).to(self.device)

        # Try fvcore first
        try:
            from fvcore.nn import FlopCountAnalysis  # ty: ignore[unresolved-import]

            flops = FlopCountAnalysis(model, dummy)
            return flops.total() / 1e9  # GFLOPs
        except Exception:
            pass

        # Fallback: thop
        try:
            from thop import profile as thop_profile

            flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            return flops / 1e9
        except Exception:
            pass

        return None

    def _measure_latency(self, model: torch.nn.Module) -> dict[str, float]:
        """Measure inference latency."""
        dummy = torch.randn(*self.input_size).to(self.device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model(dummy)

        # Synchronize if CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                _ = model(dummy)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                times.append((end - start) * 1000)  # ms

        times = np.array(times)
        return {
            "latency_mean_ms": float(times.mean()),
            "latency_std_ms": float(times.std()),
            "latency_p50_ms": float(np.percentile(times, 50)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "latency_p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000.0 / times.mean()),
        }

    def _measure_memory(self, model: torch.nn.Module) -> dict[str, float]:
        """Measure GPU/CPU memory usage."""
        result = {}

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            dummy = torch.randn(*self.input_size).to(self.device)
            model.eval()

            mem_before = torch.cuda.memory_allocated()
            with torch.no_grad():
                _ = model(dummy)
            mem_after = torch.cuda.max_memory_allocated()

            result["gpu_memory_allocated_mb"] = mem_before / (1024**2)
            result["gpu_memory_peak_mb"] = mem_after / (1024**2)
            result["gpu_memory_inference_mb"] = (mem_after - mem_before) / (1024**2)

        # CPU memory via psutil
        try:
            import psutil

            process = psutil.Process()
            result["cpu_memory_rss_mb"] = process.memory_info().rss / (1024**2)
        except ImportError:
            pass

        return result

    def profile_pytorch_model(
        self,
        model: torch.nn.Module,
        model_name: str = "unknown",
    ) -> dict:
        """
        Full profiling of a PyTorch model.

        Args:
            model: PyTorch model
            model_name: descriptive name

        Returns:
            Dict with all profiling results
        """
        model = model.to(self.device).eval()

        result: dict[str, Any] = {"model": model_name, "device": str(self.device)}

        # Parameters
        params = self._count_parameters(model)
        result.update(params)

        # FLOPs
        gflops = self._measure_flops(model)
        if gflops is not None:
            result["gflops"] = gflops

        # Latency
        latency = self._measure_latency(model)
        result.update(latency)

        # Memory
        memory = self._measure_memory(model)
        result.update(memory)

        self.results.append(result)
        self._print_result(result)

        return result

    def _measure_gpu_memory_yolo(self, yolo) -> float:
        """Peak/resident GPU memory (MB) for one forward pass; 0.0 on CPU.

        CUDA: true peak via ``max_memory_allocated`` (after a reset). MPS: no peak
        counter exists, so read ``current_allocated_memory`` immediately after a
        forward while the output is still referenced (weights + activations).
        """
        det = yolo.model
        dev = self.device
        try:
            det.to(dev).eval()
            dummy = torch.randn(*self.input_size, device=dev)
        except Exception:
            return 0.0

        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = det(dummy)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024**2)

        if dev.type == "mps" and getattr(torch, "mps", None) is not None:
            getattr(torch.mps, "empty_cache", lambda: None)()
            with torch.no_grad():
                out = det(dummy)  # keep `out` alive so the allocation is observable
            current = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
            del out
            return current / (1024**2)

        return 0.0

    def profile_ultralytics_model(
        self,
        weights: str,
        model_name: str = "unknown",
    ) -> dict:
        """
        Profile an Ultralytics model (YOLO/RT-DETR).

        Args:
            weights: path to .pt weights
            model_name: descriptive name
        """
        from ultralytics import YOLO

        yolo = YOLO(weights)

        # Basic info
        result: dict[str, Any] = {
            "model": model_name,
            "weights": weights,
            "device": str(self.device),
        }

        # Parameters
        if hasattr(yolo.model, "model"):
            pytorch_model = yolo.model.model
        else:
            pytorch_model = yolo.model

        params = self._count_parameters(cast("torch.nn.Module", pytorch_model))
        result.update(params)
        # Canonical key consumed by phase 06 (`parameters`); total_params kept too.
        result["parameters"] = params["total_params"]
        result["params_m"] = params["total_params_m"]

        # FLOPs (GFLOPs) via Ultralytics' helpers; emitted as `flops_G` for phase 06.
        # NOTE: get_flops needs the DetectionModel (`yolo.model`, which carries
        # `.stride`), NOT the inner `yolo.model.model` Sequential — the latter returns
        # 0. Fall back to the torch-profiler variant if thop is unavailable.
        imgsz = self.input_size[-1]
        flops_g = 0.0
        try:
            from ultralytics.utils.torch_utils import get_flops

            flops_g = float(get_flops(yolo.model, imgsz=imgsz))
            if flops_g == 0.0:
                from ultralytics.utils.torch_utils import get_flops_with_torch_profiler

                flops_g = float(get_flops_with_torch_profiler(yolo.model, imgsz=imgsz))
        except Exception:
            flops_g = 0.0
        result["flops_G"] = flops_g

        # Latency measurement using predict
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(self.num_warmup):
            _ = yolo.predict(dummy_img, verbose=False)

        times = []
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = yolo.predict(dummy_img, verbose=False)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times = np.array(times)
        result.update(
            {
                "latency_mean_ms": float(times.mean()),
                "latency_std_ms": float(times.std()),
                "latency_p50_ms": float(np.percentile(times, 50)),
                "latency_p95_ms": float(np.percentile(times, 95)),
                "fps": float(1000.0 / times.mean()),
            }
        )

        # Memory — emit canonical `gpu_memory_mb` (phase 06) plus the peak alias.
        # Must be measured around a forward pass with the output tensors still alive:
        # reading after `predict` returns (as before) reports ~0 because the GPU
        # tensors have already been freed. CUDA uses its peak counter; MPS has no
        # peak API, so we read the live allocation (weights + activations) right
        # after a forward while the output is referenced.
        gpu_mem_mb = self._measure_gpu_memory_yolo(yolo)
        result["gpu_memory_mb"] = float(gpu_mem_mb)
        result["gpu_memory_peak_mb"] = float(gpu_mem_mb)

        self.results.append(result)
        self._print_result(result)

        return result

    def save_results(self, output_path: str = "experiments/baseline/profiling.csv"):
        """Save profiling results."""
        out_path = Path(output_path)
        ensure_dir(out_path.parent)

        df = pd.DataFrame(self.results)
        df.to_csv(out_path, index=False)

        json_path = out_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info("Profiling results saved to %s", out_path)
        return df

    def _print_result(self, result: dict):
        """Pretty-print profiling result."""
        logger.info(
            "%s: params=%.2fM, latency=%.2f±%.2f ms, fps=%.1f",
            result["model"],
            result.get("total_params_m", 0),
            result.get("latency_mean_ms", 0),
            result.get("latency_std_ms", 0),
            result.get("fps", 0),
        )
        if "gpu_memory_peak_mb" in result:
            logger.info("  GPU Memory: %.1f MB", result["gpu_memory_peak_mb"])


def compare_ensemble_vs_large(
    ensemble_results: list[dict],
    large_model_results: list[dict],
) -> pd.DataFrame:
    """
    Create a comparison table: ensemble vs large models.
    """
    rows = []

    for r in ensemble_results:
        rows.append({**r, "type": "ensemble"})
    for r in large_model_results:
        rows.append({**r, "type": "large_model"})

    df = pd.DataFrame(rows)

    logger.info("COMPARISON: Ensemble vs Large Models")

    cols = ["model", "type", "total_params_m", "fps", "latency_mean_ms"]
    if "mAP50" in df.columns:
        cols.append("mAP50")
    if "gpu_memory_peak_mb" in df.columns:
        cols.append("gpu_memory_peak_mb")

    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile detection models")
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="experiments/baseline/profiling.csv")
    args = parser.parse_args()

    profiler = ResourceProfiler(
        device=args.device,
        num_iterations=args.iterations,
    )

    for weights, name in zip(args.weights, args.names):
        profiler.profile_ultralytics_model(weights, name)

    profiler.save_results(args.output)
