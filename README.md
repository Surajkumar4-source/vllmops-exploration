# Exploring vLLM and vLLM-Ops on NVIDIA DGX A100

## Overview

This repository documents my ongoing exploration of **vLLM and vLLM-Ops** for deploying, managing, and testing large language model (LLM) inference on GPU infrastructure.

The objective is to understand how LLM serving works, how multiple inference instances can run concurrently, and how vLLM-Ops can help manage model lifecycle operations and observe running services.

This is an experimental learning project. The current results represent initial functional and continuous-inference testing, not a maximum-performance benchmark.

## Test Environment

| Component | Configuration |
|---|---|
| GPU server | NVIDIA DGX A100 |
| GPU count | 8 × NVIDIA A100-SXM4 |
| GPU memory | 40 GB per GPU |
| Total GPU memory | 320 GB nominal |
| Inference engine | vLLM |
| Model used | `Qwen/Qwen2.5-0.5B-Instruct` |
| Orchestration tool | vLLM-Ops |
| Inference API | OpenAI-compatible API |
| Test duration | Approximately 5 minutes |
| Test type | Continuous, concurrent inference |

## What I Have Explored

### 1. vLLM installation and baseline inference

- Set up a dedicated Python virtual environment for vLLM.
- Deployed a baseline Qwen2.5-0.5B-Instruct inference server on GPU 0.
- Tested model availability, API responses, and text generation.
- Used the OpenAI-compatible chat completions endpoint.

### 2. Exploring vLLM-Ops

- Installed and initialized vLLM-Ops.
- Created model configuration files for individual GPU instances.
- Configured separate endpoints and GPU assignments.
- Started and monitored model-serving processes.
- Explored model status, health checks, logs, and the terminal user interface (TUI).
- Tested model lifecycle operations, including starting and stopping instances.

Seven inference instances were managed through vLLM-Ops on GPUs 1–7. The baseline server on GPU 0 was launched separately.

### 3. Parallel inference across all eight GPUs

Deployed eight inference endpoints using the same Qwen2.5-0.5B-Instruct model. Each endpoint was assigned a dedicated GPU and port.

| GPU | Endpoint alias | Port |
|---:|---|---:|
| 0 | `baseline-qwen` | 8101 |
| 1 | `qwen-managed` | 8102 |
| 2 | `qwen-gpu2` | 8103 |
| 3 | `qwen-gpu3` | 8104 |
| 4 | `qwen-gpu4` | 8105 |
| 5 | `qwen-gpu5` | 8106 |
| 6 | `qwen-gpu6` | 8107 |
| 7 | `qwen-gpu7` | 8108 |

A concurrent inference test successfully generated responses from all eight endpoints.

### 4. Continuous inference testing

Ran a continuous-inference workload for approximately five minutes, submitting repeated requests concurrently across the eight GPU endpoints.

The test collected request counts, errors, generated-token counts, throughput, and end-to-end request latency.

## Initial Benchmark Results

**Test duration:** 300.62 seconds

| Metric | Observed result |
|---|---:|
| GPUs tested | 8 |
| Total successful requests | 40,582 |
| Failed requests | 0 |
| Aggregate request throughput | 134.99 requests/sec |
| Total generated output tokens | 5,193,606 |
| Aggregate output throughput | 17,276.31 tokens/sec |
| Overall average latency | 0.473 seconds |
| Overall P95 latency | 0.490 seconds |

### Per-GPU Results

| GPU | Successful requests | Output tokens | Observed output throughput |
|---:|---:|---:|---:|
| 0 | 5,040 | 645,017 | 2,145.62 tokens/sec |
| 1 | 5,013 | 641,605 | 2,134.27 tokens/sec |
| 2 | 5,130 | 656,540 | 2,183.95 tokens/sec |
| 3 | 5,100 | 652,754 | 2,171.36 tokens/sec |
| 4 | 5,094 | 651,951 | 2,168.69 tokens/sec |
| 5 | 4,976 | 636,758 | 2,118.15 tokens/sec |
| 6 | 5,132 | 656,714 | 2,184.53 tokens/sec |
| 7 | 5,097 | 652,267 | 2,169.74 tokens/sec |

All eight endpoints completed requests successfully during the recorded test, with zero reported request failures.

These figures describe the specific workload, model, serving configuration, and concurrency used in this experiment. They should not be interpreted as the maximum capability of the DGX A100 platform or as a comparison against other inference engines.

## Project Structure

The working directory currently contains the vLLM-Ops lab configuration, test scripts, and benchmark results.

```text
vllmops-lab/
├── configs/
│   └── models/
│       ├── qwen-managed.yaml
│       ├── qwen-15b-gpu2.yaml
│       ├── qwen-gpu3.yaml
│       ├── qwen-gpu4.yaml
│       ├── qwen-gpu5.yaml
│       ├── qwen-gpu6.yaml
│       └── qwen-gpu7.yaml
├── tests/
│   ├── test_all_models.py
│   ├── continuous_inference.py
│   └── continuous_inference_all_8.py
└── results/
    ├── continuous_inference_all_8_summary.json
    ├── continuous_8gpu_5min.log
    └── continuous_8gpu_2workers.log
```

*Note: Include only files that actually exist in your final GitHub commit. The GPU 2 configuration filename is historical; its current model is Qwen2.5-0.5B-Instruct, not a 15B model.*

## Current Learning Outcomes

Through this initial exploration, I have demonstrated:

- Running vLLM inference on NVIDIA A100 GPUs.
- Deploying multiple independent inference instances.
- Using vLLM-Ops to configure and manage model-serving processes.
- Sending concurrent requests to multiple GPU endpoints.
- Running sustained inference workloads and collecting performance statistics.
- Verifying endpoint health and monitoring GPU resource allocation.
- Recording benchmark results in a machine-readable JSON file.

## Next Steps

This exploration will continue with:

- Serving different model variants and comparing their behavior.
- Testing model switching and lifecycle management through vLLM-Ops.
- Comparing different request concurrency levels.
- Studying latency, throughput, GPU utilization, and memory consumption.
- Exploring batching, tensor parallelism, and additional serving configurations.
- Improving repeatability and documentation of benchmark experiments.

## Summary

This project is an ongoing effort to learn and evaluate **vLLM for LLM inference and vLLM-Ops for model-serving management** on an eight-GPU NVIDIA DGX A100 system.

The initial experiment successfully ran continuous inference across eight GPU endpoints, completing 40,582 requests and generating approximately 5.19 million output tokens over five minutes without reported request failures.

Further testing will focus on multiple model variants, workload scaling, operational management, and more systematic performance evaluation.
