import argparse
import asyncio
import contextlib
import csv
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None

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

from baseline import BaselineAsyncWrapper

_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima",
    "mike", "november", "oscar", "papa", "quebec", "romeo",
    "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu"
]

_PHASE_SPEC: List[Tuple[str, int, int]] = [
    ("gen_9_embed_1", 9, 1),
    ("gen_1_embed_1", 1, 1),
    ("gen_1_embed_9", 1, 9),
]


def _make_word_payload(width: int, seed: int) -> str:
    random.seed(seed)
    return " ".join(random.choices(_WORDS, k=width))


def _allocate_counts(total: int, gen_weight: int, embed_weight: int) -> Tuple[int, int]:
    weight_sum = gen_weight + embed_weight
    gen_count = int(round(total * (gen_weight / weight_sum)))
    gen_count = min(gen_count, total)
    embed_count = total - gen_count
    return gen_count, embed_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orthrus load-balancing experiment (baseline)")
    parser.add_argument("--total-requests", type=int, default=1000, help="Requests dispatched per phase")
    parser.add_argument("--phase-seconds", type=int, default=180, help="Duration budget per phase in seconds")
    parser.add_argument("--concurrency", type=int, default=128, help="Maximum concurrent in-flight requests")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for payload shuffling")
    parser.add_argument("--embed-model", type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--gen-model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--embed-gpu", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--worker-concurrency", type=int, default=32)
    parser.add_argument("--max-gen-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sampling-period", type=float, default=1.0, help="GPU telemetry sampling period seconds")
    parser.add_argument("--telemetry", action="store_true", help="Enable wrapper NVML telemetry (optional)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary (without samples)")
    parser.add_argument(
        "--util-csv",
        type=str,
        default=None,
        help="Optional CSV path for GPU utilization samples",
    )
    return parser


def _build_phase_plan(total_requests: int, seed: int) -> List[Tuple[str, List[str]]]:
    plans: List[Tuple[str, List[str]]] = []
    rng = random.Random(seed)
    for phase_name, gen_weight, embed_weight in _PHASE_SPEC:
        gen_count, embed_count = _allocate_counts(total_requests, gen_weight, embed_weight)
        phase_requests = ["gen"] * gen_count + ["embed"] * embed_count
        rng.shuffle(phase_requests)
        plans.append((phase_name, phase_requests))
    return plans


async def _gpu_sampler(
    gpu_indices: List[int],
    samples: Dict[int, List[Tuple[float, float]]],
    stop_event: asyncio.Event,
    period: float,
    start_perf: float,
) -> None:
    if pynvml is None:
        raise RuntimeError("pynvml is required for GPU sampling but is not available.")

    handles: Dict[int, Any] = {}
    for idx in gpu_indices:
        try:
            handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise RuntimeError(f"Failed to acquire NVML handle for GPU {idx}: {exc}") from exc

    try:
        while True:
            rel = time.perf_counter() - start_perf
            for idx, handle in handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    samples.setdefault(idx, []).append((rel, float(util.gpu)))
                except Exception:
                    samples.setdefault(idx, []).append((rel, float("nan")))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=period)
                break
            except asyncio.TimeoutError:
                continue
    finally:
        rel = time.perf_counter() - start_perf
        for idx, handle in handles.items():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                samples.setdefault(idx, []).append((rel, float(util.gpu)))
            except Exception:
                samples.setdefault(idx, []).append((rel, float("nan")))


async def _run_phase(
    phase_name: str,
    request_plan: List[str],
    concurrency: int,
    phase_seconds: float,
    submit_gen,
    submit_embed,
    seed_base: int,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(max(1, concurrency))
    task_kind: Dict[asyncio.Task[bool], str] = {}
    phase_start = time.perf_counter()

    async def _run_one(kind: str, payload: str) -> bool:
        async with sem:
            if kind == "gen":
                return await submit_gen(payload)
            return await submit_embed(payload)

    tasks: List[asyncio.Task[bool]] = []
    for idx, kind in enumerate(request_plan):
        payload_seed = seed_base + idx + 1
        payload = _make_word_payload(128, payload_seed)
        task = asyncio.create_task(_run_one(kind, payload))
        task_kind[task] = kind
        tasks.append(task)

    done, pending = await asyncio.wait(tasks, timeout=phase_seconds)

    timed_out: Dict[str, int] = {"gen": 0, "embed": 0}
    if pending:
        for task in pending:
            kind = task_kind.get(task, "gen")
            timed_out[kind] = timed_out.get(kind, 0) + 1
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    completed: Dict[str, int] = {"gen": 0, "embed": 0}
    failed: Dict[str, int] = {"gen": 0, "embed": 0}
    for task in done:
        kind = task_kind.get(task, "gen")
        if task.cancelled():
            failed[kind] = failed.get(kind, 0) + 1
            continue
        exc = task.exception()
        if exc is not None:
            failed[kind] = failed.get(kind, 0) + 1
            continue
        if task.result():
            completed[kind] = completed.get(kind, 0) + 1
        else:
            failed[kind] = failed.get(kind, 0) + 1

    duration = time.perf_counter() - phase_start
    return {
        "phase": phase_name,
        "duration_sec": duration,
        "total_dispatched": len(tasks),
        "completed": completed,
        "failed": failed,
        "timed_out": timed_out,
    }


async def _run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if pynvml is None:
        raise RuntimeError("pynvml package is required to record GPU utilization.")

    nvml_initialized = False
    try:
        pynvml.nvmlInit()
        nvml_initialized = True
    except Exception as exc:
        raise RuntimeError("Failed to initialize NVML for GPU sampling") from exc

    random.seed(args.seed)

    num_gpus = args.num_gpus
    gen_gpus = [g for g in range(num_gpus) if g != args.embed_gpu]
    if not gen_gpus:
        gen_gpus = [args.embed_gpu]

    wrapper = BaselineAsyncWrapper(
        embed_gpu=args.embed_gpu,
        gen_gpus=gen_gpus,
        embed_model=args.embed_model,
        gen_model=args.gen_model,
        enable_embed=True,
        enable_gen=True,
        enable_gpu_telemetry=args.telemetry,
    )
    await wrapper.start()

    bench_start = time.perf_counter()
    gpu_indices = sorted({args.embed_gpu} | set(gen_gpus))
    gpu_samples: Dict[int, List[Tuple[float, float]]] = {}
    sampler_stop = asyncio.Event()
    sampler_task = asyncio.create_task(
        _gpu_sampler(gpu_indices, gpu_samples, sampler_stop, args.sampling_period, bench_start)
    )

    latencies_gen_ms: List[float] = []
    latencies_embed_ms: List[float] = []

    async def _submit_gen(payload: str) -> bool:
        start = time.perf_counter()
        outs = await wrapper.generate([payload], temperature=args.temperature, max_tokens=args.max_gen_tokens)
        if outs and outs[0] is not None:
            latencies_gen_ms.append((time.perf_counter() - start) * 1000.0)
            return True
        return False

    async def _submit_embed(payload: str) -> bool:
        start = time.perf_counter()
        vecs = await wrapper.embed([payload])
        if vecs and vecs[0] is not None:
            latencies_embed_ms.append((time.perf_counter() - start) * 1000.0)
            return True
        return False

    phase_plans = _build_phase_plan(args.total_requests, args.seed)
    phase_summaries: List[Dict[str, Any]] = []
    phase_markers: List[Dict[str, float]] = []

    try:
        for phase_idx, (phase_name, request_plan) in enumerate(phase_plans):
            phase_start_rel = time.perf_counter() - bench_start
            phase_result = await _run_phase(
                phase_name,
                request_plan,
                args.concurrency,
                float(args.phase_seconds),
                _submit_gen,
                _submit_embed,
                seed_base=args.seed * 100000 + phase_idx * 1000,
            )
            phase_end_rel = time.perf_counter() - bench_start
            phase_markers.append({
                "phase": phase_name,
                "start_sec": phase_start_rel,
                "end_sec": phase_end_rel,
            })
            phase_summaries.append(phase_result)
    finally:
        sampler_stop.set()
        await sampler_task
        await wrapper.stop()
        if nvml_initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    summary = {
        "phases": phase_summaries,
        "phase_markers": phase_markers,
        "gpu_samples": {
            gpu: samples for gpu, samples in sorted(gpu_samples.items())
        },
        "latencies_gen_ms": latencies_gen_ms,
        "latencies_embed_ms": latencies_embed_ms,
    }

    if args.util_csv:
        with open(args.util_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "time_sec", "gpu_index", "util_percent"])
            for gpu_idx, samples in sorted(gpu_samples.items()):
                for rel_time, util in samples:
                    phase_name = "post"
                    for marker in phase_markers:
                        if marker["start_sec"] <= rel_time <= marker["end_sec"]:
                            phase_name = marker["phase"]
                            break
                    writer.writerow([phase_name, f"{rel_time:.3f}", gpu_idx, f"{util:.3f}"])

    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = asyncio.run(_run_benchmark(args))

    if args.json:
        print(json.dumps({
            "phases": summary["phases"],
            "phase_markers": summary["phase_markers"],
            "gpu_sample_counts": {gpu: len(samples) for gpu, samples in summary["gpu_samples"].items()},
        }))
    else:
        print("=== Load Balancing Experiment (Baseline) ===")
        for phase in summary["phases"]:
            print(
                f"{phase['phase']}: dispatched={phase['total_dispatched']} "
                f"completed_gen={phase['completed'].get('gen', 0)} completed_embed={phase['completed'].get('embed', 0)} "
                f"timed_out_gen={phase['timed_out'].get('gen', 0)} timed_out_embed={phase['timed_out'].get('embed', 0)} "
                f"duration={phase['duration_sec']:.2f}s"
            )
        for gpu_idx, samples in summary["gpu_samples"].items():
            valid = [util for _, util in samples if not math.isnan(util)]
            util_avg = sum(valid) / len(valid) if valid else float("nan")
            print(
                f"GPU {gpu_idx}: samples={len(samples)} avg_util={util_avg:.2f}%"
                if samples else f"GPU {gpu_idx}: samples=0"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
