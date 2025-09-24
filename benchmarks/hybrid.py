import argparse
import asyncio
import contextlib
import logging
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_USAGE_STATS", "0")

logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)
for _name in [
    "vllm",
    "vllm.engine",
    "vllm.core",
    "vllm.worker",
    "asyncio",
    "uvicorn",
    "httpx",
    "urllib3",
    "numexpr",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

from vllm import AsyncLLMEngine, SamplingParams
from vllm.config import PoolerConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams

from transformers import AutoTokenizer

try:
    import pynvml
except Exception:
    pynvml = None


@dataclass
class HybridSpec:
    gpu_index: int
    model_id: str
    lora_path: Optional[str]
    lora_adapter_id: Optional[str]
    pooling_type: str
    max_model_len: int
    gpu_mem_util: float
    enable_embed: bool
    enable_gen: bool


class HybridAsyncWrapper:
    """Hybrid-mode wrapper mirroring BaselineAsyncWrapper APIs."""

    def __init__(
        self,
        *,
        embed_gpu: int,
        gen_gpus: Optional[List[int]] = None,
        embed_model: Optional[str],
        gen_model: Optional[str],
        lora_path: Optional[str] = None,
        lora_adapter_id: Optional[str] = None,
        pooling_type: str = "MEAN",
        max_model_len: int = 4096,
        gpu_mem_util: float = 0.90,
        enable_embed: bool = True,
        enable_gen: bool = True,
    ) -> None:
        if not enable_embed and not enable_gen:
            raise ValueError("HybridAsyncWrapper requires embedding or generation to be enabled.")

        model_candidates = [m for m in (embed_model, gen_model) if m is not None]
        if not model_candidates:
            raise ValueError("At least one of embed_model or gen_model must be provided.")
        model_id = model_candidates[0]
        for m in model_candidates[1:]:
            if m != model_id:
                raise ValueError("HybridAsyncWrapper expects embed and gen model ids to match.")

        self.spec = HybridSpec(
            gpu_index=embed_gpu,
            model_id=model_id,
            lora_path=lora_path,
            lora_adapter_id=lora_adapter_id,
            pooling_type=pooling_type,
            max_model_len=max_model_len,
            gpu_mem_util=gpu_mem_util,
            enable_embed=enable_embed,
            enable_gen=enable_gen,
        )

        self._started = False
        self._engine: Optional[AsyncLLMEngine] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._pooling_params: Optional[PoolingParams] = None
        self._lora_request: Optional[LoRARequest] = None
        self._metrics: Dict[str, Any] = {
            "start_time": None,
            "stop_time": None,
            "requests": {
                "embed": {"per_gpu": {}, "latencies": []},
                "gen": {"per_gpu": {}, "latencies": []},
            },
            "gpu": {},
        }
        self._gpu_handles: Dict[int, Any] = {}
        self._gpu_poll_task: Optional[asyncio.Task[None]] = None
        self._gpu_poll_interval = 1.0
        self._request_log: List[Dict[str, Any]] = []

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("HybridAsyncWrapper.start() called more than once without stop().")

        self._metrics["start_time"] = time.time()
        self._metrics["stop_time"] = None
        self._metrics["requests"] = {
            "embed": {"per_gpu": {}, "latencies": []},
            "gen": {"per_gpu": {}, "latencies": []},
        }
        self._metrics["gpu"] = {}
        self._request_log = []

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.spec.gpu_index)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, use_fast=True)
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception:
            self._tokenizer = None

        enable_lora = (
            self.spec.lora_path is not None
            and self.spec.lora_adapter_id is not None
            and os.path.isdir(self.spec.lora_path)
        )
        if enable_lora:
            self._lora_request = LoRARequest(
                self.spec.lora_adapter_id,
                1,
                self.spec.lora_path,
            )
        else:
            self._lora_request = None

        engine_args = AsyncEngineArgs(
            model=self.spec.model_id,
            enforce_eager=True,
            gpu_memory_utilization=self.spec.gpu_mem_util,
            tensor_parallel_size=1,
            max_model_len=self.spec.max_model_len,
            task="hybrid",
            enable_lora=enable_lora,
            max_loras=1 if enable_lora else 0,
            max_lora_rank=256 if enable_lora else 0,
            enable_chunked_prefill=True,
            override_pooler_config=PoolerConfig(
                pooling_type=self.spec.pooling_type,
                normalize=False,
            ),
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._pooling_params = PoolingParams()

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.spec.gpu_index)
                self._gpu_handles[self.spec.gpu_index] = handle
                self._metrics["gpu"][self.spec.gpu_index] = {"util": [], "mem": []}
            except Exception:
                self._gpu_handles = {}

        if self._gpu_handles and self._gpu_poll_task is None:
            loop = asyncio.get_running_loop()
            self._gpu_poll_task = loop.create_task(self._poll_gpu_metrics())

        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return

        if self._gpu_poll_task is not None:
            self._gpu_poll_task.cancel()
            with contextlib.suppress(Exception):
                await self._gpu_poll_task
        self._gpu_poll_task = None

        engine = self._engine
        self._engine = None

        if engine is not None:
            try:
                shutdown = getattr(engine, "shutdown", None)
                if shutdown is not None:
                    if asyncio.iscoroutinefunction(shutdown):
                        await shutdown()
                    else:
                        shutdown()
            except Exception:
                pass

        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

        if self._gpu_handles and pynvml is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        self._gpu_handles = {}

        self._metrics["stop_time"] = time.time()
        self._started = False

    async def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not self.spec.enable_embed:
            raise RuntimeError("Embedding is disabled for this HybridAsyncWrapper instance.")
        if not self._started or self._engine is None:
            raise RuntimeError("HybridAsyncWrapper must be started before calling embed().")
        if self._pooling_params is None:
            raise RuntimeError("Pooling parameters not initialized; call start().")

        async def one(text: str) -> Optional[List[float]]:
            rid = str(uuid.uuid4())
            start_perf = time.perf_counter()
            start_wall = time.time()
            final_output = None
            try:
                async for out in self._engine.encode(
                    text,
                    self._pooling_params,
                    rid,
                    lora_request=self._lora_request,
                ):
                    final_output = out
            except Exception:
                latency = time.perf_counter() - start_perf
                end_wall = time.time()
                self._record_request_metrics("embed", self.spec.gpu_index, latency, start_wall, end_wall)
                return None

            latency = time.perf_counter() - start_perf
            end_wall = time.time()
            self._record_request_metrics("embed", self.spec.gpu_index, latency, start_wall, end_wall)

            if final_output is None:
                return None
            return final_output.outputs

        return await asyncio.gather(*[one(text) for text in texts])

    async def generate(
        self,
        prompts: List[str],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 128,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.spec.enable_gen:
            raise RuntimeError("Generation is disabled for this HybridAsyncWrapper instance.")
        if not self._started or self._engine is None:
            raise RuntimeError("HybridAsyncWrapper must be started before calling generate().")

        async def one(prompt: str) -> str:
            rid = str(uuid.uuid4())
            start_perf = time.perf_counter()
            start_wall = time.time()
            final_output = None
            try:
                sp = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                async for out in self._engine.generate(prompt, sp, rid):
                    final_output = out
            except Exception:
                latency = time.perf_counter() - start_perf
                end_wall = time.time()
                self._record_request_metrics("gen", self.spec.gpu_index, latency, start_wall, end_wall)
                print("[hybrid] generation error", file=sys.stderr)
                return ""

            latency = time.perf_counter() - start_perf
            end_wall = time.time()
            self._record_request_metrics("gen", self.spec.gpu_index, latency, start_wall, end_wall)

            if final_output is None or not final_output.outputs:
                return ""
            try:
                return final_output.outputs[0].text or ""
            except Exception:
                return ""

        return await asyncio.gather(*[one(prompt) for prompt in prompts])

    def get_metrics(self) -> Dict[str, Any]:
        start = self._metrics.get("start_time")
        stop = self._metrics.get("stop_time") or time.time()
        runtime = None
        if start is not None:
            runtime = max(0.0, stop - start)

        summary: Dict[str, Any] = {
            "runtime_sec": runtime,
            "requests": {
                "embed": {},
                "gen": {},
            },
            "gpu": {},
        }

        for kind in ("embed", "gen"):
            req_metrics = self._metrics["requests"].get(kind, {})
            latencies = req_metrics.get("latencies", [])
            summary["requests"][kind] = {
                "count": len(latencies),
                "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else None,
                "p95_latency_ms": (
                    statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else None
                ),
                "max_latency_ms": max(latencies) * 1000 if latencies else None,
                "per_gpu": {},
            }

            per_gpu = req_metrics.get("per_gpu", {})
            for gpu_idx, entry in per_gpu.items():
                gpu_lat = entry.get("latencies", [])
                summary["requests"][kind]["per_gpu"][gpu_idx] = {
                    "count": entry.get("count", 0),
                    "avg_latency_ms": statistics.mean(gpu_lat) * 1000 if gpu_lat else None,
                    "throughput_rps": (entry.get("count", 0) / runtime) if runtime else None,
                    "latencies": list(gpu_lat),
                }

        for gpu_idx, entry in self._metrics.get("gpu", {}).items():
            util_samples = entry.get("util", [])
            mem_samples = entry.get("mem", [])
            summary["gpu"][gpu_idx] = {
                "avg_util_percent": statistics.mean(util_samples) if util_samples else None,
                "max_util_percent": max(util_samples) if util_samples else None,
                "avg_mem_percent": statistics.mean(mem_samples) if mem_samples else None,
                "max_mem_percent": max(mem_samples) if mem_samples else None,
                "sample_count": len(util_samples),
            }

        summary["request_traces"] = list(self._request_log)

        return summary

    async def _poll_gpu_metrics(self) -> None:
        try:
            while True:
                handle = self._gpu_handles.get(self.spec.gpu_index)
                if handle is None:
                    return
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    entry = self._metrics["gpu"].setdefault(self.spec.gpu_index, {"util": [], "mem": []})
                    entry.setdefault("util", []).append(float(util.gpu))
                    entry.setdefault("mem", []).append(float(mem.used) / float(mem.total) * 100.0)
                except Exception:
                    pass
                await asyncio.sleep(self._gpu_poll_interval)
        except asyncio.CancelledError:
            pass

    def _record_request_metrics(
        self,
        kind: str,
        gpu_index: int,
        latency: float,
        start_wall: float,
        end_wall: float,
    ) -> None:
        metrics = self._metrics["requests"].setdefault(kind, {"per_gpu": {}, "latencies": []})
        metrics.setdefault("latencies", []).append(latency)
        per_gpu = metrics.setdefault("per_gpu", {})
        entry = per_gpu.setdefault(gpu_index, {"latencies": [], "count": 0, "total_latency": 0.0})
        entry["latencies"].append(latency)
        entry["count"] += 1
        entry["total_latency"] += latency
        self._request_log.append(
            {
                "kind": kind,
                "gpu": gpu_index,
                "start_ts": start_wall,
                "end_ts": end_wall,
            }
        )

    def write_request_trace_csv(self, path: str) -> None:
        import csv

        fieldnames = ["kind", "start_ts", "end_ts", "gpu"]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._request_log:
                writer.writerow(row)
