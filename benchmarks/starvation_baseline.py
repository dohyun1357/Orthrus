import argparse
import asyncio
import csv
import json
import logging
import os
import random
import statistics
import sys
import time
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

from baseline import BaselineAsyncWrapper

_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima",
    "mike", "november", "oscar", "papa", "quebec", "romeo",
    "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu"
]

def _make_word_payload(width: int, seed: int) -> str:
    random.seed(seed)
    return " ".join(random.choices(_WORDS, k=width))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orthrus starvation benchmark (baseline)")
    parser.add_argument("--num-gen", type=int, default=1000, help="Number of generation requests to issue first")
    parser.add_argument("--num-embed", type=int, default=1000, help="Number of embedding requests to issue second")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for prompt construction")
    parser.add_argument("--embed-model", type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--gen-model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--embed-gpu", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON latencies")
    parser.add_argument("--trace-csv", type=str, default=None, help="Optional path to request trace CSV")
    parser.add_argument("--telemetry", action="store_true", help="Enable NVML GPU telemetry polling")
    parser.add_argument(
        "--latency-csv",
        type=str,
        default=None,
        help="Optional output CSV path for per-request latencies",
    )
    return parser


async def _run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
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

    try:
        tasks: List[asyncio.Task[Any]] = []
        prompt_seed = args.seed * 41 + 5
        for idx in range(args.num_gen):
            seed = prompt_seed + idx
            payload = _make_word_payload(128, seed)
            tasks.append(asyncio.create_task(_submit_gen(payload)))

        embed_seed = args.seed * 59 + 23
        for idx in range(args.num_embed):
            seed = embed_seed + idx
            payload = _make_word_payload(128, seed)
            tasks.append(asyncio.create_task(_submit_embed(payload)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [res for res in results if isinstance(res, Exception)]
        if errors:
            raise RuntimeError(f"Encountered {len(errors)} request failures; first error: {errors[0]!r}")
    finally:
        await wrapper.stop()

    if args.trace_csv:
        wrapper.write_request_trace_csv(args.trace_csv)

    gen_results = results[:args.num_gen]
    embed_results = results[args.num_gen:]
    gen_failures = sum(1 for res in gen_results if res is not True)
    embed_failures = sum(1 for res in embed_results if res is not True)

    summary = {
        "num_gen": args.num_gen,
        "num_embed": args.num_embed,
        "latencies_gen_ms": latencies_gen_ms,
        "latencies_embed_ms": latencies_embed_ms,
        "completed_gen": len(latencies_gen_ms),
        "completed_embed": len(latencies_embed_ms),
        "gen_failures": gen_failures,
        "embed_failures": embed_failures,
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
        print("=== Starvation Benchmark (Baseline) ===")
        print(
            f"generation requests: {summary['num_gen']} -> "
            f"completed={summary['completed_gen']} failures={summary['gen_failures']}"
        )
        print(
            f"embedding requests: {summary['num_embed']} -> "
            f"completed={summary['completed_embed']} failures={summary['embed_failures']}"
        )
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
    main()
