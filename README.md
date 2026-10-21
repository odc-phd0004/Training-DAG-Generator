# dagtrace — compute/communication DAGs of distributed training, captured on one GPU

`dagtrace` runs **TorchTitan's real trainer** (unmodified model, parallelization,
pipeline schedule) as one logical rank of an *N*-GPU job on a **single GPU**, by
replacing every collective with PyTorch's `fake` process-group backend. All compute
kernels execute for real with the exact shard shapes that rank would see. Every
collective and P2P transfer is recorded with its exact size, process group,
issuing CUDA stream and wait point.

The per-rank execution traces (PyTorch Execution Trace + Kineto) are turned into:

* a **per-rank DAG** of compute segments and communication operations,
* a **whole-job DAG** for all *N* ranks, with collectives matched across ranks,
* an **alpha-beta replay** (a baseline for your network model to replace),
* **Chakra execution traces** for ASTRA-sim,
* a **validation report** that cross-checks the DAG against the workload.

**Default (direct mode):** the DAG is built *while the step runs*. There is no profiler,
Execution Trace or Kineto, so capture costs about the same as a normal training step at
any scale.

```
TorchTitan (v0.3.0, unmodified) -> 1 GPU, fake PG, logical rank r of N
   hooks on collectives / P2P / Work.wait / wait_tensor / cuda.Event
   compute segments timed with CUDA events on their own stream
        |
        v
rank<r>/gpu_dag.step<k>.json    per-GPU DAG      (one capture per pipeline stage)
        |
        v
job_dag.compact.json            job DAG, any N   (stage templates + group partitions)
job_dag.json                    explicit job DAG (written up to --max-expand GPUs)
gpu_dag.rank<x>.json            any GPU's DAG on demand (--rank x)
validation.md                   checks
```

`MODE=raw` selects the profiler path (Execution Trace + Kineto). It measures pure kernel
time and adds op-level detail, alpha-beta simulation, Chakra/ASTRA-sim export, and the
fake-vs-real `compare` tool.

---

## 1. Setup (once)

Requirements: Linux, one NVIDIA GPU, Python ≥ 3.11, git, internet access.

```bash
tar xzf dagtrace.tar.gz && cd dagtrace
bash setup.sh                 # creates .venv/ and third_party/torchtitan (v0.3.0)
source .venv/bin/activate
bash scripts/smoke_test.sh    # ~1–2 min; must end with "0 failed"
```

`setup.sh` pins PyTorch 2.14.1, TorchTitan v0.3.0 and Chakra (fixed commit), and checks
CUDA, the GPU and driver stream ids. The PyPI PyTorch wheels are CUDA 13 builds and need
NVIDIA driver ≥ 580. With an older driver the script switches to
`TORCH_INDEX=https://download.pytorch.org/whl/cu128`. If that index doesn't carry 2.14.1,
pick another CUDA variant listed on pytorch.org.

The smoke test captures TorchTitan's debug model as an 8-GPU job with PP=2 × TP=2 ×
FSDP=2. It runs both pipeline stages, then extract, stitch, validate, simulate and Chakra
export.

## 2. Capture your workloads

```bash
bash scripts/capture_job.sh <name> <N_GPUs> [--layers K] [--gpus-per-node G] -- <torchtitan args>

# Llama-3.1 8B as a 16-GPU job, TP=2 x FSDP=8
bash scripts/capture_job.sh llama3_8b_tp2_fsdp8 16 -- \
    --module llama3 --config llama3_8b --parallelism.tensor_parallel_degree 2
```

`scripts/paper_suite.sh` contains a ready set of jobs: 8B with FSDP, HSDP, 2D, and 3D
with 1F1B/interleaved; 70B with TP×FSDP and PP×TP×FSDP.

Notes:

* **TorchTitan arguments are passed through untouched.** Anything TorchTitan accepts
  (degrees, schedules, batch sizes, activation checkpointing, compile, …) changes the
  captured job.
* The script captures **one rank per pipeline stage**. In SPMD training all ranks of a
  stage run the same program on different shards. The emulated rank is
  `pp_rank × N/PP`, which is SPMD coordinate 0.
* **Data does not matter for dense models.** By default the bundled test tokenizer and
  `c4_test` are used, so no HF downloads or gated tokenizers are needed (`--keep-data`
  disables this). Vocabulary, shapes and model size come from the model config.
* **CUDA graphs are disabled during capture.** Graph replay hides per-op kernels. This
  doesn't change the DAG, only CPU launch overhead.
* **Memory.** The process holds exactly what one rank of the real job holds. A job that
  fits an 80 GB H100 fits a 96 GB RTX PRO 6000. On a 48 GB card, lower
  `--training.local_batch_size` or use `--layers K` (truncate depth; use it to show
  per-layer linearity, not as the final DAG).

## 3. DAG format (direct mode)

**Per-GPU DAG**: `rank<r>/gpu_dag.step<k>.json`, or `gpu_dag.rank<x>.json` for any rank.

```jsonc
{"rank": 0, "world_size": 8, "step": 3, "signature": "f0e6...",
 "nodes": [
   {"id": 0, "type": "compute",    "dur_us": 812.4, "phase": "fwd",
    "ops": {"mm": 3, "silu": 1, "mul": 1}},          // "ops" only with --op-names
   {"id": 1, "type": "collective", "op": "all_gather", "bytes": 2097152, "group": "41",
    "phase": "fwd"},
   {"id": 2, "type": "p2p",        "op": "send", "bytes": 1048576, "group": "15", "peer": 4}],
 "edges": [[0, 1], [1, 2], ...],             // u must finish before v starts
 "groups": {"41": [0, 1], "15": [0, 4], ...}}  // process group -> global ranks
```

**Job DAG**: `job_dag.json`. A collective is **one vertex** shared by all its members, and
a send/recv pair is one P2P vertex.

```jsonc
{"world_size": 8, "consistency": {"ok": true, ...},
 "vertices": [
   {"id": "c:0:0", "type": "compute", "rank": 0, "dur_us": 812.4},
   {"id": "C2464", "type": "collective", "op": "all_gather", "bytes": 2097152, "ranks": [0, 1]},
   {"id": "P3380", "type": "p2p", "bytes": 1048576, "src": 0, "dst": 4}],
 "edges": [["c:0:0", "C2464"], ["C2464", "c:0:1"], ...]}
```

What each field means:

* `bytes` follows nccl-tests conventions: all-gather = output size, reduce-scatter =
  input size, all-reduce = buffer size.
* `phase` labels each vertex with where in the step it ran:
  * `bwd`: inside autograd backward, including activation-checkpoint recompute.
  * `opt`: inside `optimizer.step()`.
  * `post`: after the optimizer, e.g. loss logging.
  * `fwd`: everything else, including gradient-norm/clipping between backward and the
    optimizer.
  * `a->b`: a compute segment that spans a transition, e.g. the loss and the start of
    backward.
* `ops` (optional, `--op-names`) gives the top ATen ops by call count in the segment. It
  adds Python dispatch overhead, so capture timings separately without it.
* **Edges** are true dependencies from CUDA/NCCL stream semantics (section 4). Compute
  overlapped with a collective, as in FSDP prefetch, has no edge to it.
* `dur_us` is CUDA-event time of the segment on its stream. It includes any GPU idle gaps
  inside the segment caused by slow CPU launches; this is negligible for real model sizes.
* **Matching collectives across ranks:** members of a communicator issue its collectives
  in the same order, so the i-th collective on group G is the same vertex for all members.
  Members must agree on op and bytes; any disagreement is listed in `consistency`.

**Scale.** One capture per pipeline stage, whatever N is. `job_dag.compact.json`
(templates + group partitions) is about 1 MB even at 10k GPUs. Use `--rank x` or
`dagtrace.jobdag.gpu_dag(compact, x)` for any GPU, and `dagtrace.jobdag.expand(compact)`
for the explicit graph.

Measured during development: a 10,240-GPU job (PP2 × TP2 × replicate 640 × shard 4)
captured in about 19 s per stage and passed all checks. Full expansion gave 4.4M vertices
and 7.5M edges, consistent and acyclic, in about 7 min and 6 GB RAM.

## 3b. Outputs (raw mode, `MODE=raw`)

`traces/<name>/rank<r>/step<k>.dag.json` is the per-rank DAG
(see `examples/sample_rank0_pp2.dag.json`):

```jsonc
{ "nodes": [
   {"id":"s3","type":"COMP","phase":"fwd","dur_us":812.4,"n_ops":37,"kernels":41,"stream":7,
    "top_ops":[["aten::mm",530.2], ...]},
   {"id":"m5","type":"COMM","coll":"ALL_GATHER","bytes":16777216,"in_bytes":8388608,
    "dtype":"c10::BFloat16","axes":["tp"],"group_ranks":[0,1],"peer":null,
    "async":false,"wait":"sync","issue_stream":7,"comm_stream":"coll|41","phase":"fwd"} ...],
  "edges": [["s3","m5","issue"],["m5","s4","complete"],["s3","s4","stream"], ...],
  "summary": {...}, "signature": {"sha256": "..."} }
```

* `bytes` follows nccl-tests conventions: all-gather = output size, reduce-scatter =
  input size, all-reduce = buffer size.
* `dur_us` is GPU kernel time from Kineto (device timestamps).
* `step<k>.fine.json.gz` has the same DAG at op granularity, before chain compression.

`global.dag.json` contains all ranks' nodes (`"<rank>:<id>"`), per-rank edges, plus:

* `collectives`: one entry per collective instance with its member nodes.
* `p2p`: matched send/recv pairs.
* `consistency`: the cross-rank checks.

`global.dag.sim.json` adds `est_us` per transfer and a dependency replay
(`simulation.iteration_us`). Cluster parameters can be set with `--cluster my.json`
(`gpus_per_node`, `intra_bw_GBps`, `inter_bw_GBps`, `intra_alpha_us`, `inter_alpha_us`).

## 4. Dependency model

The edges mirror CUDA/NCCL stream semantics:

| edge | meaning |
|---|---|
| `stream` | compute on the same CUDA stream executes in issue order |
| `event` | cross-stream wait recorded from `torch.cuda.Event.record/wait`. FSDP2 overlaps all-gathers with compute through side streams and events, not `async_op`. |
| `issue` | a collective starts after the work issued before it on its issuing stream. ProcessGroupNCCL makes its stream wait on the current stream. |
| `fifo` | collectives on one communicator run in order. P2P is FIFO per (group, peer, direction), because batched isend/irecv progress concurrently. |
| `complete` | the issuing/waiting stream resumes after the collective. Sync ops resume immediately, async ops at `Work.wait()`, functional collectives at `wait_tensor`, and otherwise at the first consumer of the output. |

Compute between communication boundaries is chain-compressed into segments. This is
exact: no dependency is added or removed. `tests/test_streams.py` checks the model on a
synthetic FSDP-style trace with a side-stream all-gather overlapping compute.

## 5. Establishing trust (for the paper)

The evidence chain, strongest last:

1. **Derived, not authored.** The DAG comes from executing the unmodified framework
   (TorchTitan v0.3.0, PyTorch 2.14.1). The hooks only add profiler annotations. The
   standard PyTorch Execution Trace and Chakra schema are used.
2. **Iteration stability.** `validate` hashes the DAG structure (collectives, sizes,
   groups, edges) per profiled step. Identical hashes support the "fixed structure"
   premise empirically.
3. **Analytical cross-checks** (`validation.md`):
   * total parameter count equals the analytical count, exactly;
   * every PP transfer equals `mbs × seq/tp × dim × 2 B`, with `n_microbatches` per direction;
   * TP collectives carry the sequence-parallel activation size;
   * FSDP reduce-scatter volume is at least one full gradient pass;
   * HSDP all-reduce equals the local gradient shard.
4. **Cross-rank consistency.** Pipeline stages are captured in independent processes.
   The stitcher verifies that every collective is issued by all group members with
   identical type and size, and that every send matches a recv of equal size.
5. **Fake vs. real equivalence.** Capture the config with `MODE=raw` on your GPU, run the
   same config for real, and compare:

   ```bash
   # on a real 16-GPU allocation (2 nodes x 8), same TorchTitan args:
   torchrun --nnodes 2 --nproc_per_node 8 ... -m dagtrace.titan_capture --real --mode raw \
       --out traces/real_llama3_8b_tp2_fsdp8 -- --module llama3 --config llama3_8b \
       --parallelism.tensor_parallel_degree 2
   python -m dagtrace.extract traces/real_llama3_8b_tp2_fsdp8
   python -m dagtrace.compare traces/llama3_8b_tp2_fsdp8 traces/real_llama3_8b_tp2_fsdp8 \
       --cluster my_cluster.json
   ```

   `compare` reports four things: whether the collective sequence, node sequence and
   edges are identical; compute-time error (isolated single-GPU vs. the real run, which
   includes NCCL interference); the measured NCCL time per collective type; and the error
   of the alpha-beta comm model. A few hours on a rented 2-node machine is enough for one
   or two configurations. Report the structural identity and the timing errors. Install
   with the same `setup.sh` there. Use a shared `--out` directory, or copy every node's
   `rank<r>/` folders together before running `extract`.

   The same comparison was run during development on CPU with a real 2-process gloo
   pipeline. The fake and real DAGs were **structurally identical**: same hashes, node
   sequences and edges (`examples/cpu_fake_vs_real_pp2.json`). The timing difference there
   is CPU contention between the two processes and says nothing about GPUs.

## 6. Limitations (state them in the paper)

* **Compute times come from your GPU,** not the cluster's. Either model a cluster of
  your GPU type, rescale per kernel, or sweep the compute:comm ratio.
* **Compute is measured in isolation.** In a real run, NCCL kernels share SMs and memory
  bandwidth with overlapped compute. Quantify this with `compare` (item 5 above).
* **CPU-side gaps are not in the DAG.** Kernel launch latency and data loading are
  excluded; the DAG is GPU-centric.
* **MoE / expert parallelism is not reliable in fake mode.** Token-dispatch sizes come
  from an all-to-all of counts, which the fake backend does not perform. Dense models
  (Llama-style) with FSDP/HSDP/TP/SP/PP (1F1B, interleaved) are what has been exercised.
  Context parallel should work, but validate it with the checks.
* **Unobserved waits are inferred.** Async c10d ops without an observed `Work.wait()`
  get their wait point from the first use of their output. Waits issued through the
  generic `torch.Stream`/`torch.accelerator` API (rare) are not observed.
* **The comm model is a no-contention baseline.** It is meant to be replaced by your
  network model or ASTRA-sim (via the Chakra export).

## 7. Troubleshooting

* **`stream-id calibration mismatch`** in `validation.md`: driver stream ids did not equal
  Kineto's. Only calibrated streams are then mapped; report this with the `meta.json`.
* **`coverage.kernels` FAIL** means more than 1% of GPU time was not attributed to an op.
  Usually this is a profiler buffer overflow on huge traces. Use fewer `--profile-steps`
  or `--layers`.
* **OOM**: lower `--training.local_batch_size` or use `--layers K`.
* **The first steps are slow**: flex attention compiles with Triton on first use. That is
  why there are 3 warmup steps.
* **Logs**: `traces/<name>/capture.log`. Raw traces stay in `rank<r>/` for re-processing.

## Layout

```
dagtrace/online.py        direct-mode tracer (hooks + CUDA-event timing -> per-GPU DAG)
dagtrace/jobdag.py        per-GPU DAGs -> job DAG (compact + explicit), any-rank DAGs
dagtrace/commtag.py       raw-mode annotation hooks (collectives, Work.wait, CUDA events, stream ids)
dagtrace/titan_capture.py TorchTitan launcher (fake/real), config + fake-PP patches, profiling
dagtrace/topology.py      process-group / device-mesh partitions
dagtrace/extract.py       ET + Kineto -> per-rank DAG
dagtrace/stitch.py        per-rank -> N-rank DAG, collective matching, consistency
dagtrace/validate.py      stability + analytical + consistency report
dagtrace/simulate.py      alpha-beta comm estimates and dependency replay
dagtrace/compare.py       fake-vs-real comparison
dagtrace/to_chakra.py     Chakra/ASTRA-sim export
scripts/                  smoke_test, capture_job, paper_suite
tests/test_streams.py     unit test of the stream/event model (raw path)
tests/test_online.py      unit test of the direct tracer
tests/cpu_shim.py         CPU-only test shim (development only; never used on GPU)
```
