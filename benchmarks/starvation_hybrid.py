import argparse
import asyncio
import csv
import json
import logging
import os
import random
import statistics
import sys
from typing import Any, Dict, List

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
    parser = argparse.ArgumentParser(description="Orthrus starvation benchmark (hybrid)")
    parser.add_argument("--num-gen", type=int, default=375, help="Number of generation requests to issue first")
    parser.add_argument("--num-embed", type=int, default=250, help="Number of embedding requests to issue second")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for prompt construction")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lora-path", type=str, default="../e5-mistral-7b-instruct/lora")
    parser.add_argument("--lora-adapter-id", type=str, default="e5_adapter")
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON latencies")
    parser.add_argument("--trace-csv", type=str, default=None, help="Optional path to request trace CSV")
    parser.add_argument(
        "--latency-csv",
        type=str,
        default=None,
        help="Optional output CSV path for per-request latencies",
    )
    return parser


def _collect_latencies(metrics: Dict[str, Any], kind: str) -> List[float]:
    per_gpu = metrics.get("requests", {}).get(kind, {}).get("per_gpu", {})
    latencies: List[float] = []
    for gpu_metrics in per_gpu.values():
        latencies.extend(gpu_metrics.get("latencies", []))
    return [lat * 1000.0 for lat in latencies]


async def _run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
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
        tmp_tasks = []
        tasks: List[asyncio.Task[Any]] = []
        prompt_seed = args.seed * 37 + 11
        for idx in range(args.num_gen):
            seed = prompt_seed + idx
            payload = _make_word_payload(512, seed)
            tmp_tasks.append(wrapper.generate([payload], temperature=args.temperature, max_tokens=args.max_gen_tokens))


        embed_seed = args.seed * 53 + 17
        for idx in range(args.num_embed):
            seed = embed_seed + idx
            payload = _make_word_payload(128, seed)
            tmp_tasks.append(wrapper.embed([payload]))

        for t in tmp_tasks:
            tasks.append(asyncio.create_task(t))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [res for res in results if isinstance(res, Exception)]
        if errors:
            raise RuntimeError(f"Encountered {len(errors)} request failures; first error: {errors[0]!r}")
    finally:
        await wrapper.stop()

    if args.trace_csv:
        wrapper.write_request_trace_csv(args.trace_csv)

    metrics = wrapper.get_metrics()
    latencies_gen_ms = _collect_latencies(metrics, "gen")
    latencies_embed_ms = _collect_latencies(metrics, "embed")

    summary = {
        "num_gen": args.num_gen,
        "num_embed": args.num_embed,
        "latencies_gen_ms": latencies_gen_ms,
        "latencies_embed_ms": latencies_embed_ms,
    }

    if args.latency_csv:
        with open(args.latency_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["request_type", "latency_ms"])
            for lat in latencies_gen_ms:
                writer.writerow(["gen", lat])
            for lat in latencies_embed_ms:
                writer.writerow(["embed", lat])

    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary = asyncio.run(_run_benchmark(args))

    if args.json:
        print(json.dumps(summary))
    else:
        print("=== Starvation Benchmark (Hybrid) ===")
        print(f"generation requests: {summary['num_gen']} -> completed={len(summary['latencies_gen_ms'])}")
        print(f"embedding requests: {summary['num_embed']} -> completed={len(summary['latencies_embed_ms'])}")
        if summary['latencies_gen_ms']:
            print(
                "gen latency ms (min/p50/max): "
                f"{min(summary['latencies_gen_ms']):.2f}/"
                f"{statistics.median(summary['latencies_gen_ms']):.2f}/"
                f"{max(summary['latencies_gen_ms']):.2f}"
            )
        if summary['latencies_embed_ms']:
            print(
                "embed latency ms (min/p50/max): "
                f"{min(summary['latencies_embed_ms']):.2f}/"
                f"{statistics.median(summary['latencies_embed_ms']):.2f}/"
                f"{max(summary['latencies_embed_ms']):.2f}"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
