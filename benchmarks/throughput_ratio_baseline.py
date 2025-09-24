import argparse
import asyncio
import logging
import os
import random
import statistics
import sys
import time
from typing import Dict, List

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

from baseline import BaselineAsyncWrapper  # noqa: E402


def _make_word_payload(width: int, seed: int) -> str:
    base = "lorem"
    return " ".join(f"{base}{i%1000}" for i in range(max(1, width)))


def _comma_separated_ints(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",")]
    result: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            result.append(int(part))
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"Invalid integer value in list: {part!r}") from exc
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orthrus Fig.1 baseline throughput benchmark")
    parser.add_argument("--clients", type=int, default=512)
    parser.add_argument("--ratio-gen", type=float, default=0.5)
    parser.add_argument("--run-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--embed-model", type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--gen-model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lora-path", type=str, default="../e5-mistral-7b-instruct/lora")
    parser.add_argument("--lora-adapter-id", type=str, default="e5_adapter")
    parser.add_argument("--max-gen-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--embed-gpu", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument(
        "--embed-gpus",
        type=_comma_separated_ints,
        default="0",
        help="Comma-separated list of GPU indices to host embedding models (default auto-selects 2).",
    )
    parser.add_argument(
        "--gen-gpus",
        type=_comma_separated_ints,
        default="1,2,3",
        help="Comma-separated list of GPU indices to host generation models (default auto-selects 2).",
    )
    parser.add_argument("--telemetry", action="store_true", help="Enable NVML GPU telemetry polling")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--trace-csv", type=str, default=None)
    return parser


def _choose_task(ratio_gen: float) -> str:
    if ratio_gen <= 0.0:
        return "embed"
    if ratio_gen >= 1.0:
        return "gen"
    return "gen" if random.random() < ratio_gen else "embed"


async def _client_loop(
    client_id: int,
    stop_time: float,
    ratio_gen: float,
    wrapper: BaselineAsyncWrapper,
    counters: Dict[str, int],
    lock: asyncio.Lock,
    max_gen_tokens: int,
    temperature: float,
) -> None:
    local_counts = {"ops": 0, "gen": 0, "embed": 0}
    prompt_seed = client_id * 31 + 7
    embed_seed = client_id * 53 + 11
    while time.time() < stop_time:
        task_kind = _choose_task(ratio_gen)
        if task_kind == "gen":
            prompt_seed += 1
            payload = _make_word_payload(128, prompt_seed)
            outs = await wrapper.generate(
                [payload],
                temperature=temperature,
                max_tokens=max_gen_tokens,
            )
            if outs and outs[0] is not None:
                local_counts["gen"] += 1
        else:
            embed_seed += 1
            payload = _make_word_payload(128, embed_seed)
            vecs = await wrapper.embed([payload])
            if vecs and vecs[0] is not None:
                local_counts["embed"] += 1
        local_counts["ops"] += 1
        await asyncio.sleep(0)

    async with lock:
        for key, value in local_counts.items():
            counters[key] = counters.get(key, 0) + value


async def _run_benchmark(args: argparse.Namespace) -> Dict[str, float]:
    random.seed(args.seed)

    def _dedupe_preserve(values: List[int]) -> List[int]:
        seen = set()
        ordered: List[int] = []
        for value in values:
            if value in seen:
                continue
            ordered.append(value)
            seen.add(value)
        return ordered

    num_gpus = max(1, args.num_gpus)

    embed_gpus = list(args.embed_gpus) if args.embed_gpus else []
    if not embed_gpus:
        embed_gpus = [args.embed_gpu]
        for candidate in range(num_gpus):
            if candidate == args.embed_gpu:
                continue
            embed_gpus.append(candidate)
            if len(embed_gpus) >= 2:
                break
    embed_gpus = _dedupe_preserve(embed_gpus)
    if not embed_gpus:
        raise ValueError("At least one embedding GPU must be configured.")

    gen_gpus = list(args.gen_gpus) if args.gen_gpus else []
    if not gen_gpus:
        remaining = [g for g in range(num_gpus) if g not in embed_gpus]
        gen_gpus.extend(remaining[:2])
        if not gen_gpus:
            gen_gpus.append(embed_gpus[0])
        if len(gen_gpus) < 2 and len(embed_gpus) > 1:
            for candidate in embed_gpus[1:]:
                if candidate not in gen_gpus:
                    gen_gpus.append(candidate)
                if len(gen_gpus) >= 2:
                    break
    gen_gpus = _dedupe_preserve(gen_gpus)
    if not gen_gpus:
        raise ValueError("At least one generation GPU must be configured.")

    embed_configs = [
        {"kind": "embed", "gpu_index": gpu, "model_id": args.embed_model}
        for gpu in embed_gpus
    ]
    gen_configs = [
        {"kind": "gen", "gpu_index": gpu, "model_id": args.gen_model}
        for gpu in gen_gpus
    ]

    wrapper = BaselineAsyncWrapper(
        embed_gpu=embed_gpus[0],
        gen_gpus=gen_gpus,
        embed_model=args.embed_model,
        gen_model=args.gen_model,
        # lora_path=args.lora_path,
        # lora_adapter_id=args.lora_adapter_id,
        enable_embed=True,
        enable_gen=True,
        enable_gpu_telemetry=args.telemetry,
        embed_configs=embed_configs,
        gen_configs=gen_configs,
    )

    # await asyncio.sleep(180)

    await wrapper.start()
    # Optional: short async warmup phase to build steady state (non-blocking)
    # You can remove this if you prefer immediate benchmarking.
    # warmup_tasks = []
    # for _ in range(min(64, args.clients)):
    #     warmup_tasks.append(wrapper.embed(["warmup text"]))
    #     warmup_tasks.append(wrapper.generate(["Warmup prompt."], temperature=0.0, max_tokens=1))
    # await asyncio.gather(*warmup_tasks, return_exceptions=True)

    try:
        stop = time.time() + args.run_seconds
        counters: Dict[str, int] = {}
        lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(
                _client_loop(
                    client_id=i,
                    stop_time=stop,
                    ratio_gen=args.ratio_gen,
                    wrapper=wrapper,
                    counters=counters,
                    lock=lock,
                    max_gen_tokens=args.max_gen_tokens,
                    temperature=args.temperature,
                )
            )
            for i in range(args.clients)
        ]
        await asyncio.gather(*tasks)
    finally:
        await wrapper.stop()

    total_ops = counters.get("ops", 0)
    gen_ops = counters.get("gen", 0)
    embed_ops = counters.get("embed", 0)
    elapsed = float(args.run_seconds)
    throughput_total = total_ops / elapsed if elapsed > 0 else 0.0
    throughput_gen = gen_ops / elapsed if elapsed > 0 else 0.0
    throughput_embed = embed_ops / elapsed if elapsed > 0 else 0.0

    metrics = wrapper.get_metrics()
    traces = metrics.get("request_traces", [])
    gen_latencies = []
    embed_latencies = []
    for gpu_metrics in metrics.get("requests", {}).get("gen", {}).get("per_gpu", {}).values():
        gen_latencies.extend(gpu_metrics.get("latencies", []))
    for gpu_metrics in metrics.get("requests", {}).get("embed", {}).get("per_gpu", {}).values():
        embed_latencies.extend(gpu_metrics.get("latencies", []))

    lat_gen_ms = [lat * 1000 for lat in gen_latencies]
    lat_embed_ms = [lat * 1000 for lat in embed_latencies]

    summary = {
        "total_ops": total_ops,
        "gen_ops": gen_ops,
        "embed_ops": embed_ops,
        "throughput_total": throughput_total,
        "throughput_gen": throughput_gen,
        "throughput_embed": throughput_embed,
        "lat_p50_gen_ms": statistics.median(lat_gen_ms) if lat_gen_ms else 0.0,
        "lat_p50_embed_ms": statistics.median(lat_embed_ms) if lat_embed_ms else 0.0,
        "request_traces": traces,
    }

    if args.trace_csv:
        # Note: write_request_trace_csv pulls from internal request log already populated
        from baseline import BaselineAsyncWrapper as _B  # silence linter unused import
        # We cannot call wrapper.write_request_trace_csv after stop; but we still can since log is kept.
        # If you prefer writing before stop, move this call earlier.
        wrapper.write_request_trace_csv(args.trace_csv)

    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    results = asyncio.run(_run_benchmark(args))

    if args.csv:
        print(
            "fig1_baseline,clients,ratio_gen,run_seconds,total_ops,gen_ops,embed_ops,"
            "throughput_total,throughput_gen,throughput_embed,lat_p50_gen_ms,lat_p50_embed_ms"
        )
        print(
            f"fig1_baseline,{args.clients},{args.ratio_gen:.3f},{args.run_seconds},"
            f"{results['total_ops']},{results['gen_ops']},{results['embed_ops']},"
            f"{results['throughput_total']:.6f},{results['throughput_gen']:.6f},{results['throughput_embed']:.6f},"
            f"{results['lat_p50_gen_ms']:.2f},{results['lat_p50_embed_ms']:.2f}"
        )
    else:
        print("=== Fig.1 Baseline Benchmark ===")
        print(f"clients: {args.clients}")
        print(f"ratio_gen: {args.ratio_gen:.3f}")
        print(f"run_seconds: {args.run_seconds}")
        print(f"total ops: {results['total_ops']} (gen={results['gen_ops']} embed={results['embed_ops']})")
        print(
            "throughput rps: "
            f"total={results['throughput_total']:.3f} "
            f"gen={results['throughput_gen']:.3f} "
            f"embed={results['throughput_embed']:.3f}"
        )
        print(
            "median latency ms: "
            f"gen={results['lat_p50_gen_ms']:.2f} "
            f"embed={results['lat_p50_embed_ms']:.2f}"
        )


if __name__ == "__main__":
    main()
