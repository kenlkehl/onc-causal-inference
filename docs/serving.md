# Serving the Stage 2 language models

Stage 2 sends two kinds of requests to language models: the
[interpretation LLM](glossary.md#interpretation-llm-and-extraction-llm) reads
evidence cards, writes extraction definitions, and reviews aggregate extraction
quality, and the extraction LLM reads one patient's record at a time and
returns that patient's values. Stage 1 uses no generative LLM. This page
explains how to serve the two roles, either from OpenAI-compatible servers you
start yourself or from vLLM pools that OCI starts and stops, and how to size
requests to the servers' context windows. Every setting named here is listed in
[Configuration](configuration.md#stage2).

Each role is served independently:

| Interpretation LLM | Extraction LLM | Configuration |
|---|---|---|
| External server | External server(s) | `stage2.endpoint`; `stage2.extraction_llm.endpoint` or `.endpoints` |
| Managed vLLM pool | Managed vLLM pool | `stage2.vllm`; `stage2.extraction_llm.vllm` (see [Two managed models](#two-managed-models)) |
| Either | The other | Any combination of the above |

## External OpenAI-compatible servers

Any server that implements the OpenAI chat-completions API and a `/models`
listing works, for example a vLLM server started with `vllm serve MODEL`.

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "",
    "workers": 16,
    "extraction_llm": {
      "endpoint": "http://127.0.0.1:8020/v1",
      "model": "",
      "workers": 32
    }
  }
}
```

**Model resolution.** At startup Stage 2 queries each endpoint's `/models`
list. An empty `model` is resolved only when the server advertises exactly one
model ID; otherwise set it explicitly. An explicit `model` must be among the
advertised IDs, matched exactly. Every replica in a pool must report the same
backing model. The resolved identities are written to
`stage2/model_identity.json` and checked whenever the run resumes (see
[Checkpoints and resuming](configuration.md#checkpoints-and-resuming)).

**API keys.** `stage2.api_key` and `stage2.extraction_llm.api_key` default to
`EMPTY`. When a key is absent from the configuration (or is the `<redacted>`
placeholder that saved `run_config.json` files contain), Stage 2 reads
`OCI_STAGE2_API_KEY` (interpretation), `OCI_STAGE2_EXTRACTION_API_KEY`
(extraction), or `OCI_STAGE2_EXTRACTION_RUNTIME_API_KEY` (extraction
continuation route). To relaunch a
[saved run](glossary.md#fresh-run-and-saved-run) against servers that require a
key, set these environment variables or pass the keys with `--set`.

### Several extraction servers

To spread extraction across several equivalent servers, list them in
`stage2.extraction_llm.endpoints` instead of `endpoint`:

```json
{
  "stage2": {
    "extraction_llm": {
      "model": "MODEL",
      "workers": 128,
      "endpoints": [
        {"endpoint": "http://SERVER_A:8002/v1", "max_concurrency": 64},
        {"endpoint": "http://SERVER_B:8002/v1", "max_concurrency": 64}
      ]
    }
  }
}
```

- All servers must serve the same model; Stage 2 verifies each one. The model
  can be omitted when the first server advertises exactly one model.
- `max_concurrency` caps active requests to one server; `workers` caps the
  whole pool and defaults to the sum of the caps. The effective ceiling is the
  smaller of the two, and it includes repair and retry calls.
- Requests go to the server with the lowest load score: in-flight requests
  relative to capacity, weighted by recent response time. When a vLLM server
  exposes `/metrics`, its running and waiting requests from other workloads
  count too, and the score rises steeply above 80% KV-cache use. Metrics are
  polled every 5 seconds and expire after 15 seconds; without them, local
  counts are used.
- After a transport failure a server is skipped for a cooldown that grows from
  2 to 30 seconds with repeated failures, then receives one recovery probe.
- Changing server URLs or caps preserves extraction checkpoints.

The same list can be passed on the command line with
`--stage2-extraction-endpoints '[{"endpoint": "...", "max_concurrency": 64}]'`
or to `run_synthetic.sh` as `STAGE2_EXTRACTION_ENDPOINTS`.

### Continuation route

`stage2.extraction_llm.runtime_endpoint` and `runtime_model` (set together)
send the remaining extraction requests to a different server and model, while
checkpoints keep the configured extraction model as their identity.
Measurements extracted through the route record the model that produced them. When a continuation route is
set, a configured managed extraction pool is not started. It cannot be combined
with `endpoints`.

### Model families and request controls

Stage 2 detects the model family from the backing model ID that `/models`
reports (falling back to the served ID): names containing `qwen3`, `gemma4`, or
`lfm25` (ignoring punctuation and case) select the Qwen 3, Gemma 4, or LFM 2.5
family, and names containing `nemotron-3.5` (with `-`, `_`, a space, or nothing
between the parts, and `.` or `_` in the version) select Nemotron 3.5, for
example `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`. Names
containing `beam-1` as a separate word (with `-`, `_`, a space, or nothing
between the parts, and optionally a minor version such as `1.0`) select
Reflection's Beam 1, for example `reflection/Beam-1.0-501B-A23B-NVFP4`; a
served alias such as `beam` resolves through the backing model ID. Earlier
Nemotron releases, other Beam versions, and anything else are treated
generically. The family determines:

- **Thinking controls.** Recognized families receive the thinking switch through
  `chat_template_kwargs.enable_thinking` (LFM 2.5 also `enableThinking`).
  Qwen 3, Gemma 4, and LFM 2.5 also get a short instruction added to the
  prompt in case the switch is rejected. Nemotron 3.5 receives only the
  chat-template switch its model card documents, with `medium_effort: true`
  for reasoning effort `low` or `medium`, and never `reasoning_effort`.
  Beam 1 always reasons, so it has no switch: it receives the level as
  `reasoning_effort`, as vLLM's Beam recipe documents, or, if the server
  rejects that field, as `chat_template_kwargs.reasoning_effort`, with
  `none` sent as `minimal`, its lowest level. Every Beam request names a level,
  because its chat template reasons at its maximum level (`max`) when a request
  names none. Generic models receive only `reasoning_effort`.
- **Reasoning effort.** `stage2.interpretation_reasoning_effort` (default
  `auto`) resolves to `high` (`xhigh` for Qwen 3.8 Flash Next).
  `stage2.extraction_reasoning_effort` (default `none`) turns thinking off for
  initial extraction requests (`minimal` for Beam 1). Repairs after the fifth
  turn thinking on.
- **Sampling.** Each family has a checked-in sampling profile
  ([Sampling profiles](configuration.md#sampling-profiles)).

**Request negotiation.** If a server rejects an optional control (HTTP 400 or
422 naming, for example, `reasoning_effort`, `chat_template_kwargs`,
`response_format`, `top_k`, or `repetition_penalty`), Stage 2 retries the same
request without it, ending with a plain request that has no JSON schema hint or
extra fields. Each variant uses one HTTP attempt from `transport_max_attempts`
(default 6), so a server that rejects many controls can exhaust the attempt
budget; raise `transport_max_attempts` for such servers. Qwen 3.8 Flash Next is
strict: only the JSON response-format hint may be dropped, and rejection of its
reasoning or sampling controls fails the request. Beam 1 may move its reasoning
level from the top-level field to the template variable and may drop the JSON
hint or rejected sampling controls, but no Beam request omits the level.

Timeouts, retries, and repairs are listed in
[Request and recovery defaults](configuration.md#request-and-recovery-defaults).

## Context budgeting

Stage 2 never truncates clinical text or evidence to fit a context window.
Prompts are sized in advance where the token count can be computed, and
otherwise an oversized prompt is rejected by the server and the request fails.

**Interpretation prompts.** The window is the smallest of the `max_model_len`
that the server advertises in `/models` and any `--max-model-len` in
`stage2.vllm.extra_args`. Prompts are counted with the served model's
tokenizer, reserving room for the larger of the thinking and non-thinking chat
templates, with a 4,096-token margin. For an external server, budgeting needs
the model's tokenizer to load locally (from the Hugging Face cache or by
download). If it cannot, Stage 2 logs once per model:

```text
Prompt token budgeting is disabled for external endpoint model 'MODEL' because its tokenizer could not be loaded locally (...). The server will reject prompts that exceed its context window.
```

and continues without token budgeting. The character limits
(`stage2.max_prompt_chars` and the other `*_max_prompt_chars` settings) always
apply.

**Extraction prompts.** The extraction window is the configured
`stage2.extraction_context_window_tokens` (default 131,072); it is not read from
the server. Set it to the extraction server's real window. If it is larger, the
server rejects long requests with HTTP 400 and the run fails. Each request's
input budget is

```text
extraction_context_window_tokens − max(extraction_max_tokens, extraction_reasoning_max_tokens) − extraction_context_margin_tokens
```

and a long record is split into ordered chunks of at most
`extraction_chunk_size_tokens` source tokens that fit that budget. The output
ceiling of each request is then lowered to whatever the window leaves. Token
counts use the extraction model's own tokenizer, loaded from the managed pool's
`download_dir` or the Hugging Face cache. If that tokenizer cannot be loaded,
the run stops:

```text
Stage 2 cannot enforce prompt token limits because the tokenizer 'MODEL' could not be loaded. Make the tokenizer available in the vLLM download directory or the Hugging Face cache.
```

Changing the extraction window, margin, chunk size, output ceilings, or model on
a saved run re-extracts the affected chunks.

## Managed vLLM pools

With `stage2.vllm` (interpretation) or `stage2.extraction_llm.vllm`
(extraction), OCI starts vLLM servers when Stage 2 begins, checks that they are
ready, routes requests to them, and stops them when Stage 2 finishes or is
interrupted. Stage 1 completes before any server starts.

### Installation

Managed pools need vLLM in the same Python environment:

```bash
uv sync --frozen --extra local-llm
.venv/bin/python scripts/configure_local_cuda.py
```

The `local-llm` extra installs vLLM and the CUDA 13.0 compiler toolkit matching
the locked PyTorch and vLLM runtime. `scripts/configure_local_cuda.py` exposes
that compiler, its headers, and its runtime libraries inside `.venv`, which
FlashInfer and PyTorch need to compile GPU kernels at startup (on glibc 2.42 or
later it also patches a CUDA header declaration). Rerun it after recreating the
environment. Inside the environment, `nvcc --version` should report 13.0;
explicit `CUDA_HOME` or `CUDA_PATH` settings take precedence over the
environment's toolkit. The host needs an NVIDIA driver that supports CUDA 13.0
(R580 or later). `run_synthetic.sh` performs both setup steps automatically
when a managed pool is configured.

### Pool settings

```json
{
  "stage2": {
    "model": "MODEL_A",
    "vllm": {"gpus": "cuda:0,cuda:1,cuda:2,cuda:3", "gpus_per_server": 1},
    "extraction_llm": {
      "model": "MODEL_B",
      "vllm": {"gpus": [4, 5, 6, 7], "gpus_per_server": 1}
    }
  }
}
```

`model` is required for a managed role. The same keys apply under
`stage2.vllm` and `stage2.extraction_llm.vllm`; unknown keys are errors.

| Key | Default | Meaning |
|---|---|---|
| `gpus` | required | Logical CUDA indices (`0` or `cuda:0`), as a list or comma-separated string; interpreted after any `CUDA_VISIBLE_DEVICES` mapping |
| `server_count` | number of GPUs ÷ `gpus_per_server` | Replicas; must divide the GPUs evenly |
| `gpus_per_server` | GPUs ÷ `server_count` | Tensor-parallel width of each replica |
| `host` | `127.0.0.1` | Bind address |
| `base_port` | 8010 (interpretation), 8110 (extraction) | First public port; replicas use consecutive ports |
| `ports` | none | Explicit public ports, one per replica |
| `internal_port_base` | 20000 (interpretation), 40000 (extraction) | Start of the internal vLLM port ranges; each replica uses a 128-port range |
| `startup_timeout` | 7200 | Seconds to wait for readiness |
| `startup_poll_interval` | 2 | Seconds between readiness checks |
| `shutdown_timeout` | 30 | Seconds between the stop signal and a forced kill |
| `download_dir` | empty (Hugging Face cache) | Model download directory, passed as `--download-dir`; also used to load the tokenizer |
| `reasoning_parser` | family default | Passed as `--reasoning-parser` |
| `language_model_only` | family default | `true` or `false` adds `--language-model-only` or `--no-language-model-only`; null adds neither |
| `default_chat_template_kwargs` | none | Passed as `--default-chat-template-kwargs` |
| `trust_remote_code` | `true` for Nemotron 3.5 names, otherwise `false` | `true` adds `--trust-remote-code`, which lets vLLM run code from the model repository |
| `extra_args` | `[]` | Additional vLLM command-line tokens, as a JSON list (for example `["--gpu-memory-utilization", "0.90", "--max-model-len", "128000"]`) |
| `enabled` | `true` | `false` disables the pool |

Each replica runs
`python -m vllm.entrypoints.cli.main serve MODEL --host HOST --port PORT --tensor-parallel-size N`
plus the named options, `--api-key` when the role's key is not `EMPTY`, and
`extra_args`. The served model ID equals the configured model string. For model
names containing `gemma`, OCI adds `--reasoning-parser gemma4
--language-model-only`; for names containing `qwen`, `--reasoning-parser qwen3
--language-model-only`; for Nemotron 3.5 names, `--reasoning-parser
nemotron_v3 --trust-remote-code`, as in NVIDIA's model card (set
`trust_remote_code` to `false` to leave the flag out); for Beam 1 names,
`--reasoning-parser beam`, as in vLLM's Beam recipe, which requires a vLLM
release with Beam support. Beam replicas also get
`VLLM_ENGINE_READY_TIMEOUT_S=3600` unless that variable is already set, because
the recipe reports a first start of about 10 minutes and vLLM's default limit
is 600 seconds. Hardware-specific recipe flags (`--moe-backend`,
`--kv-cache-dtype`, speculative decoding) go in `extra_args`. Set `reasoning_parser` to `""` and
`language_model_only` to null to suppress these for a model they do not fit.
Thinking is selected per request, never as a server default. `extra_args`
cannot contain flags that OCI sets itself (`--model`, `--host`, `--port`,
`--api-key`, `--served-model-name`, `--tensor-parallel-size`, `-tp`,
`--download-dir`, `--reasoning-parser`, `--language-model-only`,
`--no-language-model-only`, `--default-chat-template-kwargs`); use the named
settings instead.

### Lifecycle and files

1. Before launch, every public port is test-bound; a busy port stops the run.
2. Each replica starts in its own process group with `CUDA_VISIBLE_DEVICES`
   set to its GPUs (mapped through the parent's `CUDA_VISIBLE_DEVICES`).
3. A replica is ready when its `/models` list is non-empty. If a replica exits
   or `startup_timeout` passes first, startup fails and the last lines of its
   log are reported. The model identity check then confirms the served model.
4. On completion, interruption (SIGINT, SIGTERM, SIGHUP), or failure, OCI sends
   SIGTERM to each process group, waits `shutdown_timeout` seconds, then sends
   SIGKILL.

Pool state is recorded under `OUTPUT_DIR/stage2/vllm_servers/`:

| Path | Content |
|---|---|
| `{pool}/manifest.json` | Status (`starting`, `ready`, `failed`, `stopped`, `interrupted`), and for each replica its PID, endpoint, ports, GPUs, redacted command, log path, served model IDs, and exit code |
| `{pool}/server_NNN.log` | vLLM output, appended across launches |
| `model_phase.json` | With two managed roles: active role, allocation mode, GPU assignments, and switch history |

`{pool}` is `orchestrator` or `extractor` for pools on their configured GPUs,
`orchestrator_all_gpus` or `extractor_all_gpus` during alternation, and
`shared` when both roles use one model.

### Ports

Public ports start at `base_port` and increase by one per replica. Internal
vLLM ports start at `internal_port_base` and use 128 ports per replica. Pools
that run at the same time must use disjoint internal ranges (OCI checks this)
and distinct public ports (a clash is reported as an unavailable port when the
second pool starts). OCI does not check that two pools' GPU lists are disjoint.

### Model caching

Models download on first use into `download_dir`, or into the Hugging Face
cache when `download_dir` is empty. Set `HF_HOME` to a persistent volume before
launching to keep the cache across machines or containers. Gated models require
Hugging Face authentication in the environment (`.venv/bin/hf auth login`).
Allow disk space for the Python and CUDA environment, both LLM checkpoints, the
Stage 1 embedding model, and run artifacts.

## Two managed models

When both roles use managed pools, Stage 2 schedules the GPUs as follows.

**Same model.** When `stage2.model` equals `stage2.extraction_llm.model`, OCI
starts one pool over the union of both GPU lists and serves both roles from it
in rotation, using per-request thinking controls. The pool takes its ports,
internal port base, `extra_args`, `download_dir`, and API key from the
extraction pool's settings. `model_phase.json` records `shared_model`.

**Different models: all-GPU alternation.** Only one model is resident at a
time, spread over all GPUs of both lists:

1. The interpretation model writes the feature definitions for every outer
   fold.
2. The pool is stopped and the extraction model is started on all GPUs.
3. Whenever a fold needs the other role, in-flight requests finish, checkpoints
   are kept, the pool is replaced by the other model, and the fold resumes.

During alternation, a model keeps its configured tensor-parallel width when
that width divides the GPU count and adds replicas on consecutive ports;
otherwise it runs one replica across all GPUs. Each switch logs a
`Stage 2 failed outer_fold=...` message and records `_ManagedStage2ModelSwitch`
in the fold's `recovery_status.json`; both are expected and the fold resumes.
On resume, if extraction checkpoints exist, the run starts with the saved
active role.

**Rapid switching: configured split.** If two consecutive switches happen less
than `stage2.vllm_rapid_switch_seconds` apart (default 900 seconds, measured
from switch to switch including model load time), both models are started on
their own configured GPU lists and ports and stay resident for the rest of the
run. This choice is recorded in `model_phase.json` and kept on resume. Set
`vllm_rapid_switch_seconds` to 0 to always alternate.

## Hardware presets

`run_synthetic.sh` ships four presets in `scripts/presets/`
([Configuration](configuration.md#presets) lists the other defaults they set).
GPU indices are logical, after any `PHYSICAL_GPUS` or `CUDA_VISIBLE_DEVICES`
mapping.

| | `external` | `h100x8` | `rtxpro6000x8` | `dgx_station_gb300` |
|---|---|---|---|---|
| Hardware | Any; servers started separately | 8 × H100 80 GB | 8 × RTX PRO 6000 Blackwell 96 GB | DGX Station: 1 × GB300 (252 GB HBM3e), 72-core Grace (Arm) CPU |
| Interpretation model | `RedHatAI/Gemma-4-31B-IT-FP8-Dynamic` at `http://127.0.0.1:8010/v1` | `RedHatAI/gemma-4-31B-it-FP8-dynamic`, managed, GPUs 0–3 | `nvidia/Gemma-4-26B-A4B-NVFP4`, managed | `nvidia/Gemma-4-26B-A4B-NVFP4`, managed |
| Extraction model | `google/gemma-4-e4b-it` at `http://127.0.0.1:8020/v1` | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, managed, GPUs 4–7 | Same model as interpretation | Same model as interpretation |
| GPU scheduling | — | Two models alternate over all eight GPUs; configured split after rapid switching | One shared pool, one replica per GPU on all eight GPUs | One shared server on the GB300, after Stage 1 has used it |
| Tensor parallelism | — | 1 | 1 | 1 |
| Public ports | 8010, 8020 | 8010–8017 and 8110–8117 | 8110–8117 | 8110 |
| Internal port bases | — | 20000 and 30000 | 30000 (shared pool) | 30000 (shared server) |
| vLLM arguments | — | `--gpu-memory-utilization 0.90 --max-model-len 128000 --max-num-seqs 32` | `--gpu-memory-utilization 0.90 --max-model-len 262144 --max-num-seqs 32` | `--gpu-memory-utilization 0.90 --max-model-len 262144 --max-num-seqs 64` |
| Extraction window | 131,072 (code default; `one_conf_one_mod` uses 128,000) | 128,000, margin 4,096 | 262,144, margin 4,096 | 262,144, margin 4,096 |

For the `external` preset, start the two servers yourself before launching, or
point `STAGE2_ENDPOINT` and `STAGE2_EXTRACTION_ENDPOINT` elsewhere. With the
`rtxpro6000x8` preset, setting `STAGE2_MODEL` to a different model makes the
two models alternate, with GPUs 0–3 and 4–7 as the configured split. vLLM reads
the FP8 and NVFP4 quantization from each checkpoint. NVIDIA's NVFP4 repository
name omits `IT`, but the checkpoint is instruction tuned.

The `dgx_station_gb300` preset expects the station's optional RTX PRO display
GPU at PCI position 0 and the GB300 at position 1, the numbering `nvidia-smi`
shows. CUDA's default order (fastest first) can number them differently, so
the preset exports `CUDA_DEVICE_ORDER=PCI_BUS_ID` and sets `PHYSICAL_GPUS=1`.
It also sets `MIN_TOTAL_GPU_GB=150`, which no display GPU (at most 96 GB) can
satisfy, so selecting the wrong GPU stops the launch at detection; the check
uses total rather than free memory because a vLLM server on the GB300 leaves
Stage 1 less than that free. On a station without a display GPU, set
`PHYSICAL_GPUS=0`. Check the launcher's `GPU memory:` line for
the GB300. Stage 1 defaults to four concurrent text-model/HTR and neural-query
contexts on that GPU, each in a separate process sharing the CPU budget.
Set `STAGE1_CONTEXTS_PER_GPU=2` (or another positive integer) to change that
limit, including when using benchmark `--llm-url`. More processes use more GPU
memory; this is a concurrency limit, not automatic memory sizing. Restart the
workflow to apply it; completed contexts and checkpointed training folds remain
reusable. Individual HTR folds stay serial within each process.
The Grace CPU is Arm (aarch64): the lockfile's torch and vLLM have
aarch64 wheels, but this preset has not yet been run on a station.

A typical managed launch on an eight-GPU machine:

```bash
git clone https://github.com/kenlkehl/onc-causal-inference.git
cd onc-causal-inference
nvidia-smi                      # all eight GPUs should be listed
PRESET=h100x8 ./run_synthetic.sh one_conf_one_mod /persistent/results/one_conf_one_mod_h100x8
```

The launcher installs the environment, detects GPUs with
`scripts/detect_all_evidence_hardware.py`, runs Stage 1 on the selected GPUs,
and then starts the managed pools for Stage 2. Rerun the same command to
resume. Set `OCI_PYTHON=$PWD/.venv/bin/python` to reuse an installed
environment without synchronizing it.

`scripts/detect_all_evidence_hardware.py` prints one tab-separated line (GPU
count, devices, Stage 1 workers, Stage 2 workers, CPU count, GPU memory
summary) and requires PyTorch to see at least one CUDA GPU:

| Option | Default | Meaning |
|---|---|---|
| `--outer-folds`, `--inner-folds` | required | Fold counts, used to size Stage 1 workers |
| `--gpu-count` | `auto` (all eligible) | Number of GPUs to select; the ones with the most free memory are chosen |
| `--workers` | `auto` | Stage 1 workers; `auto` is the smaller of the CPU count and the larger of outer × (inner + 1) and the GPUs' capacity at 8 GiB per worker |
| `--stage2-workers` | 32 | Stage 2 request concurrency; `auto` caps 32 at the CPU count |
| `--min-free-vram-gib` | 20 (`run_synthetic.sh` passes `MIN_FREE_GPU_GB`, default 0) | Free memory a GPU needs to be eligible |
| `--min-total-vram-gib` | 0 (`run_synthetic.sh` passes `MIN_TOTAL_GPU_GB`, default 0) | Total memory a GPU needs to be eligible, which identifies a GPU model |
| `--stage1-vram-gib-per-worker` | 8 | GPU memory assumed per concurrent Stage 1 context |

## Troubleshooting

| Message (abbreviated) | Meaning and remedy |
|---|---|
| `Stage 2 managed vLLM port is unavailable at HOST:PORT` | A previous server, another process, or another pool holds the port. Stop it or change `base_port`/`ports` |
| `managed vLLM server N exited with code C before readiness; log=...` | vLLM failed to start: out of GPU memory, invalid arguments, a CUDA or FlashInfer compilation problem, or vLLM missing. Read `server_NNN.log` |
| `Managed Stage 2 requires vLLM in the Python environment (install .[local-llm])` | Install the `local-llm` extra |
| `timed out waiting for managed Stage 2 vLLM servers after Ns` | Slow download or kernel compilation; raise `startup_timeout` or pre-download the model |
| `SM 12.x requires CUDA >= 12.9` or `No supported CUDA architectures found` | A system `nvcc` below version 12.9 is found first on the path. Run `scripts/configure_local_cuda.py` in the environment and unset `CUDA_HOME`/`CUDA_PATH` |
| `... CUDA_VISIBLE_DEVICES hides all GPUs` or `... logical indices outside CUDA_VISIBLE_DEVICES` | `gpus` refers to devices that are not visible; indices are logical after the mapping |
| `stage2.vllm.extra_args cannot override managed argument F` | Use the named pool setting instead of the raw flag |
| `... overlapping internal rendezvous ranges` | Give concurrently resident pools distinct `internal_port_base` values |
| `Stage 2 model auto-discovery requires exactly one served model; ... Set stage2.model explicitly.` | The server lists several models |
| `Stage 2 ... endpoint .../models advertises [...], not the selected model 'M'` | The configured model ID does not match the served ID exactly (case-sensitive) |
| `Stage 2 cannot resume because the actual running model identity changed for the primary role` | A saved run must keep its interpretation model; use a different output directory for a different model |
| `Stage 2 outer fold N has feature definitions from a different evidence plan, feature-definition policy, or explicit-feature configuration` | Interpretation sampling, reasoning, output ceiling, evidence, or explicit features changed on a saved run |
| `Stage 2 endpoint rejected optional request controls; trying compatibility variant i/n` | Informational: request negotiation is in progress |
| `Stage 2 transport exhausted N attempt(s); ...` or `... exhausted its HTTP attempt budget while negotiating optional request controls` | The server kept failing or rejected too many controls. Check the server; raise `transport_max_attempts` for servers that reject many controls |
| `Stage 2 server stopped the response with finish_reason=length` | The output ceiling was reached; the request is repaired. Persistent cases need a larger window or ceiling |
| `Stage 2 extraction context window leaves no input-token budget` or `serial extraction prompt leaves no room for one source character` | `extraction_context_window_tokens` is too small for the output ceilings and margin; reduce `extraction_feature_batch_size` or `extraction_max_tokens`, or raise the window to the server's real value |
| HTTP 400 from the extraction server on long records | `extraction_context_window_tokens` exceeds the server's window |
| `Stage 2 cannot enforce prompt token limits because the tokenizer ... could not be loaded` | Make the extraction model's tokenizer available in `download_dir` or the Hugging Face cache (and authenticate for gated models) |
