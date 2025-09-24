import argparse
import asyncio
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
from hybrid import HybridAsyncWrapper  # noqa: E402
from dummy_lora import DummyLoRAResult, generate_dummy_lora  # noqa: E402

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


def _comma_separated_ints(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",")]
    out: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid integer value inside list: {part!r}") from exc
    return out


def _comma_separated_floats(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",")]
    out: List[float] = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid float value inside list: {part!r}") from exc
    return out


def _dedupe_preserve(values: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


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
    wrapper,
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


def _summarize_counters(
    counters: Dict[str, int],
    run_seconds: float,
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    total_ops = counters.get("ops", 0)
    gen_ops = counters.get("gen", 0)
    embed_ops = counters.get("embed", 0)

    throughput_total = total_ops / run_seconds if run_seconds > 0 else 0.0
    throughput_gen = gen_ops / run_seconds if run_seconds > 0 else 0.0
    throughput_embed = embed_ops / run_seconds if run_seconds > 0 else 0.0

    gen_latencies = []
    embed_latencies = []
    for gpu_metrics in metrics.get("requests", {}).get("gen", {}).get("per_gpu", {}).values():
        gen_latencies.extend(gpu_metrics.get("latencies", []))
    for gpu_metrics in metrics.get("requests", {}).get("embed", {}).get("per_gpu", {}).values():
        embed_latencies.extend(gpu_metrics.get("latencies", []))

    lat_gen_ms = [lat * 1000 for lat in gen_latencies]
    lat_embed_ms = [lat * 1000 for lat in embed_latencies]

    return {
        "total_ops": float(total_ops),
        "gen_ops": float(gen_ops),
        "embed_ops": float(embed_ops),
        "throughput_total": throughput_total,
        "throughput_gen": throughput_gen,
        "throughput_embed": throughput_embed,
        "lat_p50_gen_ms": statistics.median(lat_gen_ms) if lat_gen_ms else 0.0,
        "lat_p50_embed_ms": statistics.median(lat_embed_ms) if lat_embed_ms else 0.0,
    }


async def _run_baseline_once(
    *,
    clients: int,
    run_seconds: int,
    ratio_gen: float,
    embed_model: str,
    gen_model: str,
    embed_gpus: Sequence[int],
    gen_gpus: Sequence[int],
    max_gen_tokens: int,
    temperature: float,
    telemetry: bool,
) -> Dict[str, float]:
    embed_configs = [
        {"kind": "embed", "gpu_index": gpu, "model_id": embed_model}
        for gpu in embed_gpus
    ]
    gen_configs = [
        {"kind": "gen", "gpu_index": gpu, "model_id": gen_model}
        for gpu in gen_gpus
    ]

    wrapper = BaselineAsyncWrapper(
        embed_gpu=embed_gpus[0],
        gen_gpus=list(gen_gpus),
        embed_model=embed_model,
        gen_model=gen_model,
        enable_embed=True,
        enable_gen=True,
        enable_gpu_telemetry=telemetry,
        embed_configs=embed_configs,
        gen_configs=gen_configs,
    )

    await wrapper.start()
    try:
        stop = time.time() + run_seconds
        counters: Dict[str, int] = {}
        lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(
                _client_loop(
                    client_id=i,
                    stop_time=stop,
                    ratio_gen=ratio_gen,
                    wrapper=wrapper,
                    counters=counters,
                    lock=lock,
                    max_gen_tokens=max_gen_tokens,
                    temperature=temperature,
                )
            )
            for i in range(clients)
        ]
        await asyncio.gather(*tasks)
    finally:
        await wrapper.stop()

    metrics = wrapper.get_metrics()
    return _summarize_counters(counters, float(run_seconds), metrics)


async def _run_hybrid_once(
    *,
    clients: int,
    run_seconds: int,
    ratio_gen: float,
    model_id: str,
    lora: Optional[DummyLoRAResult],
    max_gen_tokens: int,
    temperature: float,
) -> Dict[str, float]:
    wrapper = HybridAsyncWrapper(
        embed_gpu=0,
        embed_model=model_id,
        gen_model=model_id,
        lora_path=str(lora.path) if lora else None,
        lora_adapter_id=lora.adapter_id if lora else None,
    )

    await wrapper.start()
    try:
        stop = time.time() + run_seconds
        counters: Dict[str, int] = {}
        lock = asyncio.Lock()
        tasks = [
            asyncio.create_task(
                _client_loop(
                    client_id=i,
                    stop_time=stop,
                    ratio_gen=ratio_gen,
                    wrapper=wrapper,
                    counters=counters,
                    lock=lock,
                    max_gen_tokens=max_gen_tokens,
                    temperature=temperature,
                )
            )
            for i in range(clients)
        ]
        await asyncio.gather(*tasks)
    finally:
        await wrapper.stop()

    metrics = wrapper.get_metrics()
    return _summarize_counters(counters, float(run_seconds), metrics)


def _compute_gpu_layout(
    *,
    num_gpus: int,
    embed_gpu: int,
    embed_gpus_opt: Optional[Sequence[int]],
    gen_gpus_opt: Optional[Sequence[int]],
) -> tuple[List[int], List[int]]:
    embed_gpus = list(embed_gpus_opt) if embed_gpus_opt else []
    if not embed_gpus:
        embed_gpus = [embed_gpu]
        for candidate in range(num_gpus):
            if candidate == embed_gpu:
                continue
            embed_gpus.append(candidate)
            if len(embed_gpus) >= 2:
                break
    embed_gpus = _dedupe_preserve(embed_gpus)
    if not embed_gpus:
        raise ValueError("At least one embedding GPU must be configured.")

    gen_gpus = list(gen_gpus_opt) if gen_gpus_opt else []
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

    return embed_gpus, gen_gpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orthrus throughput benchmark suite for multiple models",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Base model ID to benchmark (generation model).")
    parser.add_argument("--variant", type=str, choices=["baseline", "hybrid", "both"],
                        default="both",
                        help="Which variant(s) to execute.")
    parser.add_argument("--embed-model", type=str,
                        help="Embedding model ID (required for baseline unless provided via scenario config).")
    parser.add_argument("--gen-model", type=str,
                        help="Optional override for generation model ID (defaults to --model).")
    parser.add_argument("--clients", type=int, default=128)
    parser.add_argument("--run-seconds", type=int, default=120)
    parser.add_argument("--ratios", type=_comma_separated_floats,
                        default=[0.1, 0.5, 0.9],
                        help="Comma-separated list of generation request ratios (e.g. 0.1,0.5,0.9).")
    parser.add_argument("--max-gen-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--embed-gpu", type=int, default=0)
    parser.add_argument("--embed-gpus", type=_comma_separated_ints, default=None)
    parser.add_argument("--gen-gpus", type=_comma_separated_ints, default=None)
    parser.add_argument("--telemetry", action="store_true",
                        help="Enable NVML GPU telemetry for baseline variant.")
    parser.add_argument("--lora-ranks", type=_comma_separated_ints, default=[8],
                        help="Comma-separated list of LoRA ranks for hybrid variant.")
    parser.add_argument("--dummy-lora-root", type=str, default="generated_lora",
                        help="Directory where synthetic LoRA adapters should be written.")
    parser.add_argument("--lora-seed", type=int, default=1234)
    parser.add_argument("--force-lora", action="store_true",
                        help="Always regenerate dummy LoRA weights even if cached.")
    parser.add_argument("--csv", action="store_true",
                        help="Emit results in CSV format instead of human readable summary.")
    return parser


def _emit_csv_header() -> None:
    print(
        "variant,model,ratio_gen,lora_rank,total_ops,gen_ops,embed_ops,"
        "throughput_total,throughput_gen,throughput_embed,lat_p50_gen_ms,lat_p50_embed_ms"
    )


def _emit_csv_row(variant: str, model: str, ratio: float, rank: Optional[int], summary: Dict[str, float]) -> None:
    rank_str = "" if rank is None else str(rank)
    print(
        f"{variant},{model},{ratio:.3f},{rank_str},"
        f"{int(summary['total_ops'])},{int(summary['gen_ops'])},{int(summary['embed_ops'])},"
        f"{summary['throughput_total']:.6f},{summary['throughput_gen']:.6f},{summary['throughput_embed']:.6f},"
        f"{summary['lat_p50_gen_ms']:.2f},{summary['lat_p50_embed_ms']:.2f}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    gen_model = args.gen_model or args.model
    embed_model = args.embed_model

    variants: Iterable[str]
    if args.variant == "both":
        variants = ("baseline", "hybrid")
    else:
        variants = (args.variant,)

    if args.csv:
        _emit_csv_header()

    for variant in variants:
        if variant == "baseline":
            if not embed_model:
                parser.error("--embed-model is required when running the baseline variant.")
            embed_gpus, gen_gpus = _compute_gpu_layout(
                num_gpus=max(1, args.num_gpus),
                embed_gpu=args.embed_gpu,
                embed_gpus_opt=args.embed_gpus,
                gen_gpus_opt=args.gen_gpus,
            )
            for ratio in args.ratios:
                summary = asyncio.run(_run_baseline_once(
                    clients=args.clients,
                    run_seconds=args.run_seconds,
                    ratio_gen=ratio,
                    embed_model=embed_model,
                    gen_model=gen_model,
                    embed_gpus=embed_gpus,
                    gen_gpus=gen_gpus,
                    max_gen_tokens=args.max_gen_tokens,
                    temperature=args.temperature,
                    telemetry=args.telemetry,
                ))
                if args.csv:
                    _emit_csv_row("baseline", args.model, ratio, None, summary)
                else:
                    print("=== Baseline ===")
                    print(f"model: {args.model}")
                    print(f"ratio_gen: {ratio:.3f}")
                    print(f"total ops: {int(summary['total_ops'])} (gen={int(summary['gen_ops'])} embed={int(summary['embed_ops'])})")
                    print(
                        "throughput rps: "
                        f"total={summary['throughput_total']:.3f} "
                        f"gen={summary['throughput_gen']:.3f} "
                        f"embed={summary['throughput_embed']:.3f}"
                    )
                    print(
                        "median latency ms: "
                        f"gen={summary['lat_p50_gen_ms']:.2f} "
                        f"embed={summary['lat_p50_embed_ms']:.2f}"
                    )
        elif variant == "hybrid":
            root = Path(args.dummy_lora_root)
            for ratio in args.ratios:
                for rank in args.lora_ranks:
                    lora_result = generate_dummy_lora(
                        args.model,
                        rank=rank,
                        root=root,
                        seed=args.lora_seed + rank,
                        force=args.force_lora,
                    )
                    summary = asyncio.run(_run_hybrid_once(
                        clients=args.clients,
                        run_seconds=args.run_seconds,
                        ratio_gen=ratio,
                        model_id=args.model,
                        lora=lora_result,
                        max_gen_tokens=args.max_gen_tokens,
                        temperature=args.temperature,
                    ))
                    if args.csv:
                        _emit_csv_row("hybrid", args.model, ratio, rank, summary)
                    else:
                        print("=== Hybrid (LoRA) ===")
                        print(f"model: {args.model} rank={rank}")
                        print(f"lora path: {lora_result.path}")
                        print(f"ratio_gen: {ratio:.3f}")
                        print(f"total ops: {int(summary['total_ops'])} (gen={int(summary['gen_ops'])} embed={int(summary['embed_ops'])})")
                        print(
                            "throughput rps: "
                            f"total={summary['throughput_total']:.3f} "
                            f"gen={summary['throughput_gen']:.3f} "
                            f"embed={summary['throughput_embed']:.3f}"
                        )
                        print(
                            "median latency ms: "
                            f"gen={summary['lat_p50_gen_ms']:.2f} "
                            f"embed={summary['lat_p50_embed_ms']:.2f}"
                        )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    main()
