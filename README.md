# Exploring vLLM and vLLM-Ops on NVIDIA DGX A100

## 1\. Overview

This repository documents an ongoing hands-on exploration of **vLLM and vLLM-Ops for deploying, managing, monitoring, and benchmarking large language model (LLM) inference workloads** on NVIDIA GPU infrastructure.

The experiments were conducted on an NVIDIA DGX A100 system equipped with eight NVIDIA A100 40 GB GPUs. The primary objective was to understand how LLM inference services operate in a multi-GPU environment, how independent model-serving instances behave under concurrent workloads, and how orchestration tooling can simplify model lifecycle management and operational visibility.

The exploration progressed through two primary test scenarios:

- **Test Case 1 — Same model across eight GPUs:** Deploying eight independent inference endpoints using Qwen2.5-0.5B-Instruct, with one model-serving process assigned to each GPU.
- **Test Case 2 — Eight distinct models across eight GPUs:** Deploying different LLMs simultaneously, with each GPU hosting a separate model endpoint, and evaluating their sustained inference behavior under concurrent request generation.

The experiments included initial model deployment, API validation, service health monitoring, lifecycle operations, concurrent inference, continuous workload generation, and throughput and latency analysis.

The results demonstrate that the DGX A100 environment successfully supported multiple independent inference services and sustained concurrent requests during the recorded test windows.


## 2\. Objectives

The primary goal was to develop a practical understanding of multi-model LLM serving on GPU infrastructure rather than simply deploying a single model and testing an isolated API endpoint.

The exploration focused on the following objectives.

### 2.1 Understand the vLLM inference-serving architecture

- Install and configure vLLM in an isolated Python environment.
- Load and serve instruction-tuned language models.
- Understand the role of the inference server in processing incoming requests and generating tokens.
- Validate model availability through OpenAI-compatible API endpoints.
- Observe how an individual model-serving process utilizes its assigned GPU.

### 2.2 Explore vLLM-Ops for model lifecycle management

- Initialize vLLM-Ops and define individual model configurations.
- Configure multiple model-serving instances with separate ports and GPU assignments.
- Start and stop model instances through the management interface.
- Inspect service status, health information, process logs, and terminal user interface (TUI) output.
- Understand how an orchestration layer can simplify management when multiple inference endpoints run on the same host.

### 2.3 Evaluate independent inference instances

- Deploy multiple instances of the same model on separate GPUs.
- Run different models concurrently on the same GPU server.
- Verify that individual endpoints can accept requests independently.
- Observe differences in request completion rates and generated-token throughput across models.

### 2.4 Measure sustained inference behavior

- Generate continuous workloads against all deployed endpoints.
- Measure successful requests, request failures, output-token counts, aggregate throughput, and latency.
- Compare the behavior of small and larger models under the configured workload.
- Identify potential areas for performance tuning and further investigation.

---

## 3\. Test Environment

### 3.1 Hardware and software configuration

| Component | Configuration |
| --- | --- |
| GPU server | NVIDIA DGX A100 |
| GPU architecture | NVIDIA A100-SXM4 |
| GPU count | 8 |
| Memory per GPU | 40 GB per GPU |
| Aggregate nominal GPU memory | 320 GB |
| Inference engine | vLLM |
| Model orchestration | vLLM-Ops |
| API interface | OpenAI-compatible HTTP API |
| Primary test model | Qwen/Qwen2.5-0.5B-Instruct |
| Test duration | Approximately 300 seconds per benchmark |
| Workload type | Continuous, concurrent inference |
| Primary metrics | Requests/sec, output tokens/sec, latency, failures |

The system provides eight independent GPU devices, allowing separate model-serving processes to be assigned to individual GPUs.

Although the aggregate nominal GPU memory is 320 GB, memory is distributed across eight physical devices. It should not be treated as one automatically pooled 320 GB allocation available to a single model instance. In these experiments, the primary deployment pattern used one GPU per independent model-serving instance.

### 3.2 Software environment

The initial setup used a dedicated Python virtual environment to isolate the vLLM installation and its Python dependencies.

The main software components were:

- **vLLM:** The inference engine responsible for model loading, request processing, and text generation.
- **vLLM-Ops:** The operational management layer explored for configuring, starting, stopping, and monitoring model-serving instances.
- **OpenAI-compatible API:** The interface used by the benchmark client to submit chat completion requests.
- **Continuous inference test client:** The workload generator used to issue repeated concurrent requests and collect benchmark statistics.
---


# Test Case 1: Same Model Across All Eight GPUs

## 5\. Experimental Objective

The first experiment investigated whether eight independent vLLM instances serving the same model could run concurrently on the DGX A100 and sustain a continuous stream of inference requests.

The model selected was `Qwen/Qwen2.5-0.5B-Instruct`.

This relatively small instruction-tuned model provided a starting point for validating the deployment architecture, API accessibility, independent GPU assignment, and concurrent request execution before introducing larger or more diverse models.

The experiment was primarily intended to validate functional correctness and establish an initial throughput baseline.

## 6\. Deployment and Initial Validation

### 6.1 Establishing a baseline inference server

The initial deployment involved:

1. Creating a dedicated Python virtual environment.
2. Installing and configuring vLLM.
3. Launching Qwen2.5-0.5B-Instruct on GPU 0.
4. Exposing the model through an OpenAI-compatible API endpoint.
5. Sending test requests to the chat completions API.
6. Verifying that the model returned valid generated responses.

The baseline server served as the first working reference point for the remaining experiments.

The validation established that the model could be loaded, the endpoint could accept requests, and inference could be performed through the API.

### 6.2 Exploring vLLM-Ops

After validating the baseline deployment, vLLM-Ops was installed and initialized to explore its model management capabilities.

The activities included:

- Creating model configuration files for individual instances.
- Defining GPU assignments and endpoint ports.
- Starting model-serving processes.
- Checking model status and health information.
- Inspecting logs and terminal user interface output.
- Testing model start and stop operations.

Seven model instances were managed through vLLM-Ops on GPUs 1–7. The GPU 0 baseline server remained independently launched.

This setup provided a practical opportunity to examine the operational differences between managing one manually launched server and coordinating multiple instances through a management layer.

## 7\. Endpoint Configuration

Each GPU hosted an independent endpoint serving the same model.

| GPU | Endpoint alias | Port | Model |
| --- | --- | --- | --- |
| 0 | `baseline-qwen` | 8101 | Qwen2.5-0.5B-Instruct |
| 1 | `qwen-managed` | 8102 | Qwen2.5-0.5B-Instruct |
| 2 | `qwen-gpu2` | 8103 | Qwen2.5-0.5B-Instruct |
| 3 | `qwen-gpu3` | 8104 | Qwen2.5-0.5B-Instruct |
| 4 | `qwen-gpu4` | 8105 | Qwen2.5-0.5B-Instruct |
| 5 | `qwen-gpu5` | 8106 | Qwen2.5-0.5B-Instruct |
| 6 | `qwen-gpu6` | 8107 | Qwen2.5-0.5B-Instruct |
| 7 | `qwen-gpu7` | 8108 | Qwen2.5-0.5B-Instruct |

This configuration separated the API endpoints and associated serving processes, allowing requests to be directed to individual GPU-backed services.

Using separate ports also made it possible to validate each instance independently before running the combined workload.

## 8\. Continuous Concurrent Inference

Once all eight endpoints were operational, a continuous-inference workload was executed for approximately five minutes.

The test client repeatedly submitted inference requests concurrently to the eight endpoints. The objective was to observe whether the services could maintain request processing over a sustained period rather than simply complete a small number of isolated API calls.

The recorded metrics included:

- Successful request count.
- Failed request count.
- Total generated output tokens.
- Aggregate request throughput.
- Aggregate output-token throughput.
- Average end-to-end request latency.
- P95 end-to-end request latency.
- Per-GPU request and token statistics.

The experiment completed without any reported request failures.

## 9\. Initial Benchmark Results

### 9.1 Aggregate results

| Metric | Observed result |
| --- | --- |
| Test duration | 300.62 seconds |
| GPUs tested | 8 |
| Successful requests | 40,582 |
| Failed requests | 0 |
| Aggregate request throughput | 134.99 requests/sec |
| Total generated output tokens | 5,193,606 |
| Aggregate output throughput | 17,276.31 tokens/sec |
| Overall average latency | 0.473 seconds |
| Overall P95 latency | 0.490 seconds |

The observed aggregate request rate was approximately 135 requests per second across the eight endpoints. The aggregate output-token rate was approximately 17.28 thousand tokens per second.

The results show that the configured workload was processed successfully by all eight endpoints during the measurement window.

### 9.2 Per-GPU results

| GPU | Successful requests | Output tokens | Output throughput |
| --- | --- | --- | --- |
| 0 | 5,040 | 645,017 | 2,145.62 tokens/sec |
| 1 | 5,013 | 641,605 | 2,134.27 tokens/sec |
| 2 | 5,130 | 656,540 | 2,183.95 tokens/sec |
| 3 | 5,100 | 652,754 | 2,171.36 tokens/sec |
| 4 | 5,094 | 651,951 | 2,168.69 tokens/sec |
| 5 | 4,976 | 636,758 | 2,118.15 tokens/sec |
| 6 | 5,132 | 656,714 | 2,184.53 tokens/sec |
| 7 | 5,097 | 652,267 | 2,169.74 tokens/sec |

The per-GPU results were relatively consistent, with output throughput ranging from approximately 2,118 to 2,185 tokens per second.

The difference between the highest and lowest recorded per-GPU throughput was approximately 3.1% of the highest value. This suggests that the eight instances delivered broadly similar results under this particular workload.

Small variations may result from differences in request completion timing, generated response lengths, scheduling, warm-up effects, or other runtime conditions. Additional repeated trials would be needed to establish whether these differences are statistically meaningful.

### 9.3 Interpretation

The first test case demonstrated several important operational outcomes:

- All eight endpoints could serve the same model concurrently.
- The API endpoints remained functional throughout the recorded workload.
- The deployment achieved approximately 135 successful requests per second in aggregate.
- The measured output throughput was broadly balanced across the eight GPUs.
- No request failures were recorded during the test.

The results provide an initial baseline for replicated-model inference on the DGX A100.

However, these measurements should not be interpreted as the maximum throughput of vLLM or the A100 hardware. The observed performance depends on the request distribution, prompt length, output length, concurrency, batching behavior, model configuration, and other serving parameters.

---

<br>

## Snapshots

<br>

## 1. 

<img width="1680" height="858" alt="Screenshot 2026-10-22 124415" src="https://github.com/user-attachments/assets/4428255b-96e8-4d64-8869-8200d88a4e87" />


<br>

## 2. 

<img width="1014" height="871" alt="Screenshot 2026-10-22 124449" src="https://github.com/user-attachments/assets/3d7d70c4-bb46-4228-b0c5-a18fbc4d9a66" />


<br>

## 3. 

<img width="1717" height="880" alt="Screenshot 2026-10-22 120917" src="https://github.com/user-attachments/assets/ca1b5116-fdc0-437a-b0c2-17dad9afa254" />



#    ____________________________________________


# Test Case 2: Eight Distinct Models Across All Eight GPUs

## 10\. Experimental Objective

The second experiment extended the investigation from replicated-model serving to a heterogeneous multi-model deployment.

Instead of loading the same model on every GPU, eight different models were served concurrently, with one model endpoint assigned to each GPU.

This scenario is closer to a shared inference environment in which several applications or teams require access to different language models at the same time.

The main objective was to evaluate how the combined workload behaved when the models differed in parameter count, architecture, and computational requirements.

The experiment also helped investigate how model selection influences request completion rate, output-token throughput, and overall latency under a common continuous workload.

## 11\. Model Deployment Configuration

The eight deployed models were:

| Model | General characteristics |
| --- | --- |
| Qwen2.5-0.5B-Instruct | Small instruction-following model |
| TinyLlama 1.1B Chat | Compact conversational model |
| Qwen2.5-1.5B-Instruct | Small instruction-tuned model |
| Qwen2.5-3B-Instruct | Medium-small instruction-tuned model |
| Qwen2.5-Coder-1.5B-Instruct | Code-oriented instruction model |
| Phi-3 Mini | Compact general-purpose language model |
| Mistral-7B-Instruct-v0.3 | Larger instruction-tuned model |
| Qwen2.5-7B-Instruct | Larger instruction-tuned model |

The models were served simultaneously on the DGX A100, with each GPU assigned to a separate inference endpoint.

This configuration allowed the benchmark to observe differences between relatively small models and models with larger parameter counts under the same general testing environment.

Model size alone does not fully determine serving performance. Architecture, precision, context length, model implementation, prompt characteristics, and generation behavior can also influence throughput.

## 12\. Inference Methodology: Benchmark used the following configuration.

| Parameter | Value |
| --- | --- |
| Hardware | NVIDIA DGX A100 |
| GPU configuration | 8 × NVIDIA A100 40 GB |
| Deployed models | 8 distinct models |
| Model placement | One model per GPU |
| Test duration | Approximately 300 seconds |
| Workers per model | 6 , 50 , 500 |
| Total concurrent workers | 48 , 400 , 4000 |
| Maximum output tokens per request | 128 |
| API interface | OpenAI-compatible `/v1/chat/completions` |
| Local proxy bypass | Enabled for localhost requests |


The benchmark client continuously submitted requests and collected per-model and aggregate statistics.

The output-token limit of 128 constrained the maximum generated response length per request. Actual output lengths could be lower, depending on the model's responses and the request configuration.

The workload was intended to examine sustained concurrent serving under the selected configuration, rather than measure an isolated single-request latency or a theoretical hardware limit.

## 13\.  Benchmark Results

To evaluate sustained inference performance, a concurrent workload was generated against all eight independently served vLLM model endpoints on an NVIDIA DGX A100 system with 8 × A100 40 GB GPUs.

### 13.1 Per-model throughput

| Model | Successful requests | Failed | Requests/sec | Output tokens/sec |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 3,998 | 0 | 13.27 | 1,557.82 |
| TinyLlama 1.1B Chat | 3,600 | 0 | 11.94 | 1,370.39 |
| Qwen2.5-1.5B-Instruct | 3,534 | 0 | 11.73 | 1,008.45 |
| Qwen2.5-3B-Instruct | 2,009 | 0 | 6.67 | 689.91 |
| Qwen2.5-Coder-1.5B-Instruct | 4,557 | 0 | 15.12 | 945.43 |
| Phi-3 Mini | 1,557 | 0 | 5.17 | 630.28 |
| Mistral-7B-Instruct-v0.3 | 1,014 | 0 | 3.36 | 430.64 |
| Qwen2.5-7B-Instruct | 1,304 | 0 | 4.33 | 419.04 |
| **Overall** | **21,573** | **0** | **71.58** | **7,051.95** |


#### Throughput illustration

<br>

<img width="1969" height="1170" alt="vllmops_bulk_inference_throughput" src="https://github.com/user-attachments/assets/ebc316df-d393-4d51-8ff8-685b343a7e03" />



The aggregate workload completed 21,573 successful requests over approximately five minutes, with no recorded request failures.

The measured aggregate request throughput was 71.58 requests per second, while output-token throughput reached 7,051.95 tokens per second.

Compared with the first test case, this workload used a more diverse collection of models, including larger models with different computational characteristics.

The lower aggregate throughput observed in this test is consistent with the fact that the deployed model mix differs from the replicated 0.5B configuration. It should not be attributed to model diversity alone, because the workload configuration and concurrency characteristics may also differ.

### 13.2 Latency and reliability

| Metric | Observed result |
| --- | --- |
| Median request latency | 0.531 seconds |
| P95 request latency | 1.639 seconds |
| Successful requests | 21,573 |
| Failed requests | 0 |
| Aggregate output throughput | 7,051.95 tokens/sec |

The median latency of 0.531 seconds indicates that half of the recorded requests completed within approximately that duration, assuming the reported median was calculated across the complete measured request population.

The P95 latency of 1.639 seconds indicates that approximately 95% of requests completed within that duration under the benchmark's latency measurement method.

The difference between median and P95 latency shows that the workload experienced a longer-latency tail. Such variation can occur when requests differ in generation length, when larger models take longer to produce tokens, or when concurrent execution and request scheduling create variable completion times.

The precise cause cannot be established from aggregate latency statistics alone. Per-model latency distributions, prompt lengths, generated-token counts, and time-series measurements would be needed for a more detailed diagnosis.

## 14\. Key Observations

### 14.1 Smaller models delivered higher output-token throughput in this workload

Qwen2.5-0.5B-Instruct achieved the highest reported output-token throughput at 1,557.82 tokens per second.

TinyLlama 1.1B Chat also delivered relatively high throughput at 1,370.39 tokens per second.

These results are consistent with the expectation that compact models often require fewer computational resources per generated token. Nevertheless, throughput also depends on the model architecture, precision, serving configuration, and request characteristics.

### 14.2 The code-oriented model processed the most requests

Qwen2.5-Coder-1.5B-Instruct completed 4,557 requests and achieved the highest reported request throughput at 15.12 requests per second.

Its output-token throughput was 945.43 tokens per second, lower than that of the 0.5B and 1.1B models.

This distinction illustrates why request throughput and output-token throughput should be analyzed separately. A model may complete more requests per second while generating fewer tokens per request on average.

### 14.3 Larger models exhibited lower measured throughput

Mistral-7B-Instruct-v0.3 recorded 3.36 requests per second and 430.64 output tokens per second.

Qwen2.5-7B-Instruct recorded 4.33 requests per second and 419.04 output tokens per second.

These measurements reflect the behavior of the configured benchmark, not an intrinsic ranking of the models. Differences in generated response lengths, prompt lengths, architecture, precision, and inference settings can influence the observed results.

### 14.4 All eight endpoints completed requests successfully

No request failures were recorded in the bulk benchmark.

This demonstrates successful functional operation during the measured interval. However, a zero-failure result over a five-minute experiment does not establish production-level reliability, long-term stability, or resilience to GPU memory exhaustion, process crashes, network faults, or overload.

### 14.5 Aggregate throughput depended on the deployed model mix

The heterogeneous deployment achieved 7,051.95 output tokens per second, compared with 17,276.31 output tokens per second in the replicated-model test.

These results should be treated as measurements from two different workload configurations rather than a controlled head-to-head comparison. The models, worker settings, and potentially other request-generation parameters differ between the tests.

A fair performance comparison would require standardized prompt and output lengths, documented generation parameters, equivalent concurrency policies, consistent measurement boundaries, and repeated benchmark runs.

---

## 15\. Comparison of the Two Experiments

| Metric | Test Case 1: Same model | Test Case 2: Distinct models |
| --- | --- | --- |
| GPU count | 8 | 8 |
| Model configuration | Same model on all GPUs | Eight different models |
| Test duration | 300.62 seconds | Approximately 300 seconds |
| Successful requests | 40,582 | 21,573 |
| Failed requests | 0 | 0 |
| Aggregate request throughput | 134.99 requests/sec | 71.58 requests/sec |
| Aggregate output throughput | 17,276.31 tokens/sec | 7,051.95 tokens/sec |
| Latency statistics | Average and P95 | Median and P95 |

The first experiment established a baseline for replicated small-model serving. The second examined concurrent serving with a mixture of model sizes and architectures.

The first configuration achieved higher aggregate throughput in the recorded tests. However, because the workloads were not identical, the difference cannot be used to quantify the performance penalty of running different models.

A more rigorous comparison would hold the workload and request-generation methodology constant while varying only the model deployment strategy.

---

## 16\. With More Workers Snapshots






<br>





## 1. 

<img width="1575" height="513" alt="image" src="https://github.com/user-attachments/assets/892bbf1a-cf8d-4cd2-b641-f0ec8deb2297" />


<br>

## 2. 


<img width="1707" height="877" alt="Screenshot 2026-10-23 113319" src="https://github.com/user-attachments/assets/6b98a18d-6ad3-4b3d-be50-811df3ea5ea1" />


## 3. 

<img width="1710" height="856" alt="Screenshot 2026-10-23 113720" src="https://github.com/user-attachments/assets/1a19263b-fbc4-4942-8ae8-3d59b343d38e" />




## 4.

<img width="1231" height="703" alt="image" src="https://github.com/user-attachments/assets/d51b1f62-b0b8-4b6c-bd29-087c3f14848a" />


## 5.


<img width="1707" height="852" alt="Screenshot 2026-10-23 114820" src="https://github.com/user-attachments/assets/7e1b21d7-e620-4a4d-b7ca-41eecde03834" />


## 6.


<img width="1707" height="843" alt="Screenshot 2026-10-23 114837" src="https://github.com/user-attachments/assets/a8c9094c-8ef3-4c00-832a-1e386221fe80" />



## 7.

<img width="1711" height="847" alt="Screenshot 2026-10-23 114902" src="https://github.com/user-attachments/assets/fd45f612-e36b-471d-a590-3eee3bccd78e" />

## 8.

<img width="1707" height="846" alt="Screenshot 2026-10-23 114948" src="https://github.com/user-attachments/assets/927059be-350d-4fe1-b581-c096fe271325" />



## 9.

<img width="1707" height="842" alt="Screenshot 2026-10-23 115006" src="https://github.com/user-attachments/assets/506c0139-a7b5-476f-93d4-34287cd6bd83" />


<br><br>



## 17\. Future Exploration and Optimization

The current experiments establish a useful foundation for further work in inference serving, resource utilization, and operational management.

### 17.1 Optimize request concurrency

Run benchmarks with different worker counts, for example 1, 2, 4, 6, 8, and 12 workers per endpoint.

This would help identify how throughput scales with concurrency and where additional workers stop providing meaningful improvements.

Measure both throughput and latency so that increased request capacity is not evaluated independently of user-facing response time.

### 17.2 Evaluate vLLM batching and scheduling

Explore the impact of vLLM configuration parameters related to batching and sequence scheduling, including `max_num_seqs` and `max_num_batched_tokens`, where supported by the installed version.

Compare throughput and latency under the same request workload and document the exact launch arguments used for each run.

### 17.3 Test different model sizes and precisions

Extend the heterogeneous deployment to additional models and supported numerical formats.

Evaluate the trade-offs between output throughput, memory consumption, latency, and model quality rather than assuming that the fastest configuration is necessarily the best application choice.

### 17.4 Monitor GPU utilization and memory

Capture per-GPU utilization and memory consumption throughout each test. Tools such as `nvidia-smi` can provide an initial view, while time-series monitoring can help correlate resource behavior with request throughput and latency.

This would make it easier to identify underutilized GPUs, memory pressure, and possible bottlenecks.

### 17.5 Investigate time to first token and inter-token latency

For streaming workloads, separate the latency before the first generated token from the time spent generating subsequent tokens.

These metrics can help distinguish request queuing and prompt-processing delays from token-generation performance.

### 17.6 Evaluate operational resilience

Test controlled failure scenarios, including restarting an inference process, stopping a model instance, and restoring service availability.

Record restart time, health-check behavior, failed requests during the interruption, and recovery time. These experiments would provide a clearer picture of vLLM-Ops lifecycle management and operational recovery.

### 17.7 Establish a repeatable benchmark suite

Create a standardized benchmark procedure with fixed prompts, output limits, sampling parameters, warm-up periods, concurrency levels, and measurement windows.

Repeat each configuration multiple times and report the mean and variability of the results.

This would improve confidence in comparisons between models, serving settings, and deployment strategies.

### 17.8 Explore multi-GPU model serving

The current deployment primarily uses one independent GPU per endpoint. A subsequent experiment could evaluate tensor parallelism for models that benefit from distributing computation across multiple GPUs.

Such a configuration would answer a different question from the current experiments: how a single model instance behaves when its computation is distributed across several GPUs, rather than how independent model instances perform side by side.

### 17.9 Improve service-level observability

Explore metrics collection and dashboards for:

- Request rate and completed requests.
- Error rate and timeout count.
- Request queue depth.
- Time to first token and inter-token latency.
- Output-token throughput.
- GPU utilization and memory consumption.
- Model loading time and process restart events.

The objective would be to connect operational status with actual inference behavior, rather than relying exclusively on benchmark summaries.

---

## 18\. Conclusion

This exploration demonstrates a practical approach to deploying and testing multiple LLM inference services on an NVIDIA DGX A100 system using vLLM and vLLM-Ops.

The first experiment validated the operation of eight independent endpoints serving Qwen2.5-0.5B-Instruct. Under the recorded five-minute workload, the deployment completed 40,582 requests without reported failures and achieved 17,276.31 output tokens per second in aggregate.

The second experiment extended the setup to eight distinct models. Across the heterogeneous deployment, the benchmark completed 21,573 requests without recorded failures and achieved 7,051.95 output tokens per second.

Together, the experiments provided hands-on experience with model deployment, GPU assignment, endpoint configuration, lifecycle management, concurrent inference, continuous workload generation, and performance measurement.

The work also highlighted the importance of separating request throughput from token throughput and interpreting latency in the context of model size, workload characteristics, and serving configuration.

The next stage is to move from functional validation toward more systematic performance characterization by standardizing workloads, collecting GPU telemetry, testing concurrency levels, evaluating serving parameters, and conducting longer-duration experiments.

The repository therefore serves as an evolving technical record of learning and experimentation with multi-model LLM serving on NVIDIA GPU infrastructure, with a focus on reproducible measurements, operational understanding, and evidence-driven optimization.




<br><br>
<br><br>

**Measurement note:** These figures represent the reported results from this specific workload, model mix, hardware, and concurrency configuration. They should not be treated as maximum hardware capacity or as a direct comparison of model quality. Output-token throughput is based on the API-reported completion-token counts.
