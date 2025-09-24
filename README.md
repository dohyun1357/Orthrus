# Orthrus Benchmark Guide

Orthrus extends vLLM with a custom scheduler that co-serves text generation and embedding workloads on the same GPU.

## Benchmarks and Figures

All benchmark scripts live under `benchmarks/`. Each script has a `--help` flag to configure workload size, request ratios, and output paths.

- **Throughput and GPU Utilization**  
  - `benchmarks/throughput_ratio_hybrid.py` (Orthrus, single GPU)  
  - `benchmarks/throughput_ratio_baseline.py` (SplitGPU baseline)  

- **Starvation / Latency Distributions**  
  - `benchmarks/starvation_hybrid.py`  
  - `benchmarks/starvation_baseline.py`  

- **Load Balancing Across Phases**  
  - `benchmarks/load_balancing_hybrid.py`  
  - `benchmarks/load_balancing_baseline.py`  


Each script writes results as CSVs for plotting. Use the `--csv`, `--latency-csv`, or `--util-csv` flags to export the metrics used in the figures.

## Models and Adapters

- Base model: `mistralai/Mistral-7B-v0.1` (via Hugging Face Hub).  
- Embedding mode: `intfloat/e5-mistral-7b-instruct` applied through a LoRA adapter.  
- LoRA adapters must be placed under `../e5-mistral-7b-instruct/lora` or specified with `--lora-path`.
