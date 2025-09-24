import os, sys, time, asyncio, logging, contextlib, statistics, subprocess, socket
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


try:
    import pynvml
except Exception:  # pragma: no cover - optional
    pynvml = None


LOGGER = logging.getLogger(__name__)


# ---------------------------
# Server launch configuration
# ---------------------------

@dataclass
class ServerSpec:
    kind: str                 # "embed" or "gen"
    gpu_index: int
    model_id: str
    pooling_type: str = "MEAN"          # for embeddings
    max_model_len: int = 4096
    gpu_mem_util: float = 0.90
    tensor_parallel_size: int = 1


class _ServerStartupError(RuntimeError):
    pass


@dataclass
class _ServerHandle:
    spec: ServerSpec
    port: int
    process: subprocess.Popen


ServerSpecLike = Union[ServerSpec, Dict[str, Any]]


# ---------------------------
# Parent-side async wrapper
# ---------------------------

class BaselineAsyncWrapper:
    """
    Baseline wrapper (Async):
      - >=1 embed workers (each 1 GPU)
      - N gen workers (each 1 GPU; can reuse the embed GPU when N == 1)
      - Async .embed() and .generate() APIs
      - Centralized response dispatch to avoid stolen-response deadlocks
      - Concurrent consumers inside workers so vLLM can batch
    """

    def __init__(
        self,
        *,
        embed_gpu: int,
        gen_gpus: Optional[List[int]],
        embed_model: Optional[str],
        gen_model: Optional[str],
        pooling_type: str = "MEAN",
        max_model_len: int = 4096,
        gpu_mem_util: float = 0.90,
        enable_embed: bool = True,
        enable_gen: bool = True,
        enable_gpu_telemetry: bool = False,
        embed_configs: Optional[Sequence[ServerSpecLike]] = None,
        gen_configs: Optional[Sequence[ServerSpecLike]] = None,
    ) -> None:
        if not enable_embed and not enable_gen:
            raise ValueError("BaselineAsyncWrapper requires at least one of embed or gen to be enabled.")

        self.enable_embed = enable_embed
        self.enable_gen = enable_gen
        self._enable_gpu_telemetry = enable_gpu_telemetry
        self._repo_root = Path(__file__).resolve().parents[1]

        def _normalize_specs(kind: str, entries: Sequence[ServerSpecLike]) -> List[ServerSpec]:
            specs: List[ServerSpec] = []
            for raw in entries:
                if isinstance(raw, ServerSpec):
                    spec = raw
                    if spec.kind != kind:
                        spec = replace(spec, kind=kind)
                    if kind == "embed" and not spec.pooling_type:
                        spec = replace(spec, pooling_type=pooling_type)
                    specs.append(spec)
                    continue

                if not isinstance(raw, dict):
                    raise TypeError(f"Unsupported server spec type: {type(raw)!r}")

                data = dict(raw)
                data.setdefault("kind", kind)
                if data["kind"] != kind:
                    raise ValueError(f"Server spec kind {data['kind']!r} does not match expected {kind!r}.")
                data.setdefault("max_model_len", max_model_len)
                data.setdefault("gpu_mem_util", gpu_mem_util)
                data.setdefault("tensor_parallel_size", 1)
                if kind == "embed":
                    data.setdefault("pooling_type", pooling_type)
                try:
                    specs.append(ServerSpec(**data))
                except TypeError as exc:
                    raise ValueError(f"Invalid server spec configuration: {raw!r}") from exc
            return specs

        self.embed_specs: List[ServerSpec] = []
        if enable_embed:
            if embed_configs is not None:
                if len(embed_configs) == 0:
                    raise ValueError("embed_configs must contain at least one entry when enable_embed is True.")
                self.embed_specs = _normalize_specs("embed", embed_configs)
            else:
                if embed_model is None:
                    raise ValueError("embed_model must be provided when enable_embed is True.")
                self.embed_specs = [
                    ServerSpec(
                        kind="embed",
                        gpu_index=embed_gpu,
                        model_id=embed_model,
                        pooling_type=pooling_type,
                        max_model_len=max_model_len,
                        gpu_mem_util=gpu_mem_util,
                    )
                ]

        gen_targets = list(gen_gpus or []) if (enable_gen and gen_configs is None) else []
        if enable_gen:
            if gen_configs is not None:
                if len(gen_configs) == 0:
                    raise ValueError("gen_configs must contain at least one entry when enable_gen is True.")
                self.gen_specs = _normalize_specs("gen", gen_configs)
            else:
                if len(gen_targets) == 0:
                    raise ValueError("At least one generation GPU is required when enable_gen is True.")
                if gen_model is None:
                    raise ValueError("gen_model must be provided when enable_gen is True.")
                self.gen_specs = [
                    ServerSpec(
                        kind="gen",
                        gpu_index=g,
                        model_id=gen_model,
                        max_model_len=max_model_len,
                        gpu_mem_util=gpu_mem_util,
                    )
                    for g in gen_targets
                ]
        else:
            self.gen_specs = []

        # HTTP server handles
        self._embed_servers: List[_ServerHandle] = []
        self._gen_servers: List[_ServerHandle] = []
        self._embed_clients: List[httpx.AsyncClient] = []
        self._gen_clients: List[httpx.AsyncClient] = []
        self._server_processes: List[subprocess.Popen] = []
        self._http_timeout = httpx.Timeout(timeout=60.0, connect=10.0, read=None)

        # Tokenizer optional (not used here, left out to keep it lean)

        self._started: bool = False
        self._stopping: bool = False
        
        # Metrics & telemetry
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
        self._gpu_poll_task: Optional[asyncio.Task] = None
        self._gpu_poll_interval: float = 1.0
        self._request_log: List[Dict[str, Any]] = []

        # RR for gen worker selection
        self._gen_rr: int = 0
        self._embed_rr: int = 0

    def _allocate_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    async def _wait_for_server(self, port: int, timeout: float = 120.0) -> None:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0, read=2.0)) as client:
            while time.time() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
                except Exception:
                    await asyncio.sleep(0.2)
                    continue
                await asyncio.sleep(0.2)
        raise _ServerStartupError(f"Timed out waiting for vLLM server on port {port} to become ready.")

    async def _launch_server(self, spec: ServerSpec) -> _ServerHandle:
        port = self._allocate_port()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_index)
        env.setdefault("VLLM_USE_V1", "0")
        env.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        repo_vllm = self._repo_root / "vllm"
        if repo_vllm.exists():
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (f"{repo_vllm}{os.pathsep}{existing}" if existing else str(repo_vllm))

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            spec.model_id,
            "--tensor-parallel-size",
            str(spec.tensor_parallel_size),
            "--max-model-len",
            str(spec.max_model_len),
            "--gpu-memory-utilization",
            str(spec.gpu_mem_util),
            "--uvicorn-log-level",
            "error",
            "--disable-uvicorn-access-log",
        ]

        if spec.kind == "embed":
            cmd.extend(["--task", "embed"])
        else:
            cmd.extend(["--enable-chunked-prefill"])

        process = subprocess.Popen(cmd, env=env, cwd=str(self._repo_root))
        self._server_processes.append(process)

        try:
            await self._wait_for_server(port)
        except Exception:
            with contextlib.suppress(Exception):
                process.kill()
            raise

        return _ServerHandle(spec=spec, port=port, process=process)

    async def _close_http_clients(self) -> None:
        for client in self._embed_clients:
            with contextlib.suppress(Exception):
                await client.aclose()
        self._embed_clients = []

        for client in self._gen_clients:
            with contextlib.suppress(Exception):
                await client.aclose()
        self._gen_clients = []

    async def _terminate_processes(self) -> None:
        procs = list(self._server_processes)
        if not procs:
            self._server_processes = []
            return

        for proc in procs:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.terminate()

        deadline = time.time() + 10.0
        while procs and time.time() < deadline:
            procs = [p for p in procs if p.poll() is None]
            if procs:
                await asyncio.sleep(0.2)

        for proc in procs:
            with contextlib.suppress(Exception):
                proc.kill()

        for proc in self._server_processes:
            with contextlib.suppress(Exception):
                proc.wait(timeout=0.1)

        self._server_processes = []
        self._embed_servers = []
        self._gen_servers = []

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("BaselineAsyncWrapper.start() called more than once without stop().")

        self._stopping = False

        all_gpu_indices = {spec.gpu_index for spec in self.embed_specs}
        for spec in self.gen_specs:
            all_gpu_indices.add(spec.gpu_index)

        # Initialize GPU telemetry if NVML is available and enabled
        self._metrics["start_time"] = time.time()
        if self._enable_gpu_telemetry and pynvml is not None and not self._gpu_handles:
            try:
                pynvml.nvmlInit()
                for idx in all_gpu_indices:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    self._gpu_handles[idx] = handle
                    self._metrics["gpu"].setdefault(idx, {"util": [], "mem": []})
            except Exception:
                self._gpu_handles = {}
        if self._enable_gpu_telemetry and self._gpu_handles and self._gpu_poll_task is None:
            loop = asyncio.get_running_loop()
            self._gpu_poll_task = loop.create_task(self._poll_gpu_metrics())

        # Reset metrics & caches
        self._metrics["requests"]["embed"] = {"per_gpu": {}, "latencies": []}
        self._metrics["requests"]["gen"] = {"per_gpu": {}, "latencies": []}
        self._metrics["gpu"] = {idx: {"util": [], "mem": []} for idx in self._gpu_handles}
        self._metrics["stop_time"] = None
        self._request_log = []

        # Start HTTP servers
        await self._close_http_clients()
        self._server_processes = []
        self._embed_servers = []
        self._gen_servers = []
        self._embed_clients = []
        self._gen_clients = []
        self._embed_rr = 0
        self._gen_rr = 0

        try:
            if self.enable_embed and self.embed_specs:
                for spec in self.embed_specs:
                    embed_handle = await self._launch_server(spec)
                    self._embed_servers.append(embed_handle)
                    self._embed_clients.append(
                        httpx.AsyncClient(
                            base_url=f"http://127.0.0.1:{embed_handle.port}",
                            timeout=self._http_timeout,
                        )
                    )

            for spec in self.gen_specs:
                handle = await self._launch_server(spec)
                self._gen_servers.append(handle)
                client = httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{handle.port}",
                    timeout=self._http_timeout,
                )
                self._gen_clients.append(client)

            self._started = True
        except Exception:
            await self._close_http_clients()
            await self._terminate_processes()
            raise

    async def stop(self) -> None:
        if not self._started and not self._server_processes:
            return

        self._stopping = True

        await self._close_http_clients()
        await self._terminate_processes()

        # Shutdown GPU telemetry
        if self._gpu_poll_task is not None:
            self._gpu_poll_task.cancel()
            with contextlib.suppress(Exception):
                await self._gpu_poll_task
        self._gpu_poll_task = None

        if self._gpu_handles and pynvml is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        self._gpu_handles = {}

        # Reset handles so the wrapper can be started again cleanly
        self._metrics["stop_time"] = time.time()
        self._started = False

    # --------- APIs ---------

    def _pick_embed_index(self) -> int:
        if not self._embed_servers:
            raise RuntimeError("Embedding is not enabled for this BaselineAsyncWrapper instance.")
        k = self._embed_rr
        self._embed_rr = (self._embed_rr + 1) % len(self._embed_servers)
        return k

    async def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not self._embed_servers or not self._embed_clients:
            raise RuntimeError("Embedding is not enabled for this BaselineAsyncWrapper instance.")

        async def one(text: str) -> Optional[List[float]]:
            k = self._pick_embed_index()
            client = self._embed_clients[k]
            server = self._embed_servers[k]
            gpu_index = server.spec.gpu_index
            model_name = server.spec.model_id
            start_perf = time.perf_counter()
            start_wall = time.time()
            embedding: Optional[List[float]] = None

            try:
                response = await client.post(
                    "/v1/embeddings",
                    json={"model": model_name, "input": text},
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or []
                if data:
                    embedding = data[0].get("embedding")
                    if embedding is not None and not isinstance(embedding, list):
                        embedding = list(embedding)
            except Exception as exc:
                LOGGER.error("Embedding request failed: %s", exc)
                embedding = None

            latency = time.perf_counter() - start_perf
            end_wall = time.time()
            self._record_request_metrics("embed", gpu_index, latency, start_wall, end_wall)
            return embedding

        return await asyncio.gather(*[one(t) for t in texts])

    def _pick_gen_index(self) -> int:
        if not self._gen_servers:
            raise RuntimeError("Generation is not enabled for this BaselineAsyncWrapper instance.")
        k = self._gen_rr
        self._gen_rr = (self._gen_rr + 1) % len(self._gen_servers)
        return k

    async def generate(
        self,
        prompts: List[str],
        *,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 128,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        if not self._gen_servers or not self._gen_clients:
            raise RuntimeError("Generation is not enabled for this BaselineAsyncWrapper instance.")
        stops = stop or None

        async def one(prompt: str) -> str:
            k = self._pick_gen_index()
            client = self._gen_clients[k]
            server = self._gen_servers[k]

            start_perf = time.perf_counter()
            start_wall = time.time()
            text = ""

            payload: Dict[str, Any] = {
                "model": server.spec.model_id,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if stops is not None:
                payload["stop"] = stops

            try:
                response = await client.post("/v1/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                if choices:
                    text = choices[0].get("text", "") or ""
            except Exception as exc:
                LOGGER.error("Generation request failed: %s", exc)
                text = ""

            latency = time.perf_counter() - start_perf
            end_wall = time.time()
            self._record_request_metrics("gen", server.spec.gpu_index, latency, start_wall, end_wall)
            return text

        return await asyncio.gather(*[one(p) for p in prompts])

    async def _poll_gpu_metrics(self) -> None:
        try:
            while True:
                for idx, handle in self._gpu_handles.items():
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_entry = self._metrics["gpu"].setdefault(idx, {"util": [], "mem": []})
                        gpu_entry["util"].append(util.gpu)
                        if mem.total:
                            gpu_entry["mem"].append((mem.used / mem.total) * 100.0)
                        else:
                            gpu_entry["mem"].append(0.0)
                    except Exception:
                        continue
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
        req_metrics = self._metrics["requests"].setdefault(kind, {"per_gpu": {}, "latencies": []})
        req_metrics.setdefault("latencies", []).append(latency)
        per_gpu = req_metrics.setdefault("per_gpu", {})
        entry = per_gpu.setdefault(gpu_index, {"latencies": [], "count": 0, "total_latency": 0.0})
        entry["latencies"].append(latency)
        entry["count"] += 1
        entry["total_latency"] += latency
        self._request_log.append({"kind": kind, "gpu": gpu_index, "start_ts": start_wall, "end_ts": end_wall})

    def write_request_trace_csv(self, path: str) -> None:
        import csv
        fieldnames = ["kind", "start_ts", "end_ts", "gpu"]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._request_log:
                writer.writerow(row)

    def get_metrics(self) -> Dict[str, Any]:
        start = self._metrics.get("start_time")
        stop = self._metrics.get("stop_time") or time.time()
        runtime = None
        if start is not None:
            runtime = max(0.0, (stop or time.time()) - start)

        summary: Dict[str, Any] = {"runtime_sec": runtime, "requests": {"embed": {}, "gen": {}}, "gpu": {}}

        for kind in ("embed", "gen"):
            req_metrics = self._metrics["requests"].get(kind, {})
            latencies = req_metrics.get("latencies", [])
            kind_out: Dict[str, Any] = {
                "count": len(latencies),
                "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else None,
                "p95_latency_ms": (statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else None),
                "max_latency_ms": max(latencies) * 1000 if latencies else None,
                "per_gpu": {},
            }
            for gpu_idx, entry in req_metrics.get("per_gpu", {}).items():
                gpu_lat = entry.get("latencies", [])
                kind_out["per_gpu"][gpu_idx] = {
                    "count": entry.get("count", 0),
                    "avg_latency_ms": (statistics.mean(gpu_lat) * 1000) if gpu_lat else None,
                    "throughput_rps": (entry.get("count", 0) / runtime) if runtime and runtime > 0 else None,
                    "latencies": gpu_lat,  # expose raw latencies (seconds) for external analysis
                }
            summary["requests"][kind] = kind_out

        for gpu_idx, entry in self._metrics["gpu"].items():
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
 
