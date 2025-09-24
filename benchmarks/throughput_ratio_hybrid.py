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

from hybrid import HybridAsyncWrapper



def _make_word_payload(width: int, seed: int) -> str:
    base = "lorem"
    return " ".join(f"{base}{i%1000}" for i in range(max(1, width)))

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orthrus Fig.1 throughput benchmark")
    parser.add_argument("--clients", type=int, default=64, help="Number of concurrent asynchronous clients")
    parser.add_argument("--ratio-gen", type=float, default=0.5, help="Probability that a client issues generation")
    parser.add_argument("--run-seconds", type=int, default=120, help="Benchmark duration in seconds")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for workload selection")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lora-path", type=str, default="../e5-mistral-7b-instruct/lora")
    parser.add_argument("--lora-adapter-id", type=str, default="e5_adapter")
    parser.add_argument("--max-gen-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--csv", action="store_true", help="Emit CSV line instead of human-readable output")
    parser.add_argument("--trace-csv", type=str, default=None, help="Optional path to request trace CSV")
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
    wrapper: HybridAsyncWrapper,
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
            outs = await wrapper.generate([payload], temperature=temperature, max_tokens=max_gen_tokens)
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

    wrapper = HybridAsyncWrapper(
        embed_gpu=0,
        embed_model=args.model,
        gen_model=args.model,
        lora_path=args.lora_path,
        lora_adapter_id=args.lora_adapter_id,
    )
    await wrapper.start()
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
    request_traces = metrics.get("request_traces", [])

    def _collect_latency_ms(kind: str) -> List[float]:
        per_gpu = metrics.get("requests", {}).get(kind, {}).get("per_gpu", {})
        latencies = []
        for entry in per_gpu.values():
            latencies.extend(entry.get("latencies", []))
        return [lat * 1000 for lat in latencies]

    lat_embed_ms = _collect_latency_ms("embed")
    lat_gen_ms = _collect_latency_ms("gen")

    summary = {
        "total_ops": total_ops,
        "gen_ops": gen_ops,
        "embed_ops": embed_ops,
        "throughput_total": throughput_total,
        "throughput_gen": throughput_gen,
        "throughput_embed": throughput_embed,
        "lat_p50_gen_ms": statistics.median(lat_gen_ms) if lat_gen_ms else 0.0,
        "lat_p50_embed_ms": statistics.median(lat_embed_ms) if lat_embed_ms else 0.0,
        "request_traces": request_traces,
    }

    if args.trace_csv:
        wrapper.write_request_trace_csv(args.trace_csv)

    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    results = asyncio.run(_run_benchmark(args))

    if args.csv:
        print(
            "fig1,clients,ratio_gen,run_seconds,total_ops,gen_ops,embed_ops,"
            "throughput_total,throughput_gen,throughput_embed,lat_p50_gen_ms,lat_p50_embed_ms"
        )
        print(
            f"fig1,{args.clients},{args.ratio_gen:.3f},{args.run_seconds},"
            f"{results['total_ops']},{results['gen_ops']},{results['embed_ops']},"
            f"{results['throughput_total']:.6f},{results['throughput_gen']:.6f},{results['throughput_embed']:.6f},"
            f"{results['lat_p50_gen_ms']:.2f},{results['lat_p50_embed_ms']:.2f}"
        )
    else:
        print("=== Fig.1 Hybrid Benchmark ===")
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
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
