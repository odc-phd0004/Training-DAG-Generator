"""Capture per-rank execution traces of a *real* TorchTitan training job on ONE GPU.

The process runs TorchTitan's own trainer (unmodified model, parallelization
and schedule code) as one logical rank of an N-GPU job, using PyTorch's
``fake`` process-group backend for every collective.  All compute kernels run
for real on the local GPU with the exact shard shapes that rank would see;
collectives return immediately, so they show up in the trace with their exact
sizes and process groups but zero duration (their time is what the network
model / simulator supplies).

For each profiled step it writes, under ``<out>/rank<R>/``:

  step<k>.et.json       PyTorch Execution Trace (CPU op graph, tensor ids/shapes)
  step<k>.kineto.json   Kineto trace (GPU kernels with device timestamps)
  meta.json             config, versions, GPU, parallel dims, PG topology,
                        local parameter counts, pipeline stage info

Usage (single GPU, fake communication):

  python -m dagtrace.titan_capture --out traces/llama3_8b_tp4_fsdp4 \
      --ngpu 16 --pp-rank 0 -- \
      --module llama3 --config llama3_8b \
      --parallelism.tensor_parallel_degree 4

Usage (real multi-GPU validation run; launch under torchrun):

  torchrun --nproc_per_node 8 -m dagtrace.titan_capture --real --out traces/real8 -- \
      --module llama3 --config llama3_8b --parallelism.tensor_parallel_degree 4
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import socket
import sys
import time

_ARGS = None          # our CLI args
_CONFIG = None        # resolved torchtitan Trainer.Config


# --------------------------------------------------------------------------- CLI
def _parse():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        ours, theirs = argv[:i], argv[i + 1:]
    else:
        ours, theirs = argv, []
    ap = argparse.ArgumentParser(prog="dagtrace.titan_capture",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ngpu", type=int, help="logical world size (fake mode)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pp-rank", type=int, default=0,
                   help="pipeline stage rank to emulate (fake mode); the emulated "
                        "global rank is pp_rank * (ngpu / pp) i.e. SPMD coordinate 0")
    g.add_argument("--rank", type=int, help="emulate this exact global rank (fake mode)")
    ap.add_argument("--warmup", type=int, default=3, help="un-profiled warmup steps")
    ap.add_argument("--profile-steps", type=int, default=2,
                    help="number of consecutive profiled steps (>=2 lets you check "
                         "iteration-to-iteration stability)")
    ap.add_argument("--layers", type=int, default=None,
                    help="truncate the model to this many transformer layers "
                         "(layer-truncation trick for large models)")
    ap.add_argument("--keep-data", action="store_true",
                    help="keep the config's tokenizer/dataset instead of the bundled "
                         "test tokenizer + c4_test (data content does not affect the DAG)")
    ap.add_argument("--keep-cuda-graphs", action="store_true",
                    help="do NOT disable CUDA graphs (not recommended: graph replay "
                         "hides per-op kernels from the trace)")
    ap.add_argument("--mode", choices=("direct", "raw"), default="direct",
                    help="direct (default): build the compute/collective/p2p DAG while the step "
                         "runs -> rank<r>/gpu_dag.step<k>.json (fast, no profiler). "
                         "raw: also record PyTorch Execution Trace + Kineto for dagtrace.extract "
                         "(pure kernel times, op-level detail; used by validate/compare/chakra)")
    ap.add_argument("--op-names", action="store_true",
                    help="direct mode: also record the top ATen op names per compute vertex "
                         "(adds Python dispatch overhead; timings slightly pessimistic)")
    ap.add_argument("--min-segment-us", type=float, default=2.0,
                    help="direct mode: drop compute segments shorter than this (no GPU work)")
    ap.add_argument("--real", action="store_true",
                    help="real communication (launch with torchrun on real GPUs)")
    ap.add_argument("--cpu-test", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(ours)
    if not a.real and not a.ngpu:
        ap.error("--ngpu is required in fake mode")
    return a, theirs


# ---------------------------------------------------------------- config hook
def _postprocess_config(cfg):
    a = _ARGS
    tr = cfg.training
    tr.steps = a.warmup + a.profile_steps
    if not a.keep_cuda_graphs and hasattr(tr, "disable_cuda_graphs"):
        tr.disable_cuda_graphs = True
    if not a.real:
        cfg.comm.mode = "fake_backend"
    # Everything below only removes I/O side work that is not part of a step.
    if hasattr(cfg, "checkpoint") and hasattr(cfg.checkpoint, "enable"):
        cfg.checkpoint.enable = False
    if getattr(cfg, "validator", None) is not None and hasattr(cfg.validator, "enable"):
        cfg.validator.enable = False
    if getattr(cfg, "profiler", None) is not None:
        cfg.profiler.enable_profiling = False
        cfg.profiler.enable_memory_snapshot = False
    if getattr(cfg, "metrics", None) is not None:
        for k in ("enable_tensorboard", "enable_wandb"):
            if hasattr(cfg.metrics, k):
                setattr(cfg.metrics, k, False)
    if not a.keep_data:
        cfg.hf_assets_path = "./tests/assets/tokenizer"
        dl = getattr(cfg, "dataloader", None)
        if dl is not None and hasattr(dl, "dataset"):
            dl.dataset = "c4_test"
            if hasattr(dl, "dataset_path"):
                dl.dataset_path = None
    if a.layers is not None:
        m = cfg.model_spec.model
        if len(m.layers) < a.layers:
            raise SystemExit(f"--layers {a.layers} > model's {len(m.layers)} layers")
        m.layers = m.layers[: a.layers]

    # fake mode: choose which logical rank this process is
    if not a.real:
        par = cfg.parallelism
        pp = max(1, par.pipeline_parallel_degree)
        if a.ngpu % pp:
            raise SystemExit(f"ngpu {a.ngpu} not divisible by pp {pp}")
        rank = a.rank if a.rank is not None else a.pp_rank * (a.ngpu // pp)
        os.environ["RANK"] = str(rank)
    return cfg


def _install_config_hook():
    from torchtitan.config import manager as M

    orig = M.ConfigManager.parse_args

    def parse_args(self, args=None):
        global _CONFIG
        cfg = orig(self, args)
        _CONFIG = _postprocess_config(cfg)
        self.config = _CONFIG
        return _CONFIG

    M.ConfigManager.parse_args = parse_args


# ------------------------------------------------- fake-PP static stage metadata
def _install_fake_pp_patch():
    """torch.distributed.pipelining cannot do dynamic shape inference over a fake
    process group, and TorchTitan 0.3.0 does not supply static metadata.  For
    decoder LLMs every stage boundary carries the hidden state, so we supply it.
    If any of this is wrong, the pipelining runtime validates every tensor it
    sends/receives against this metadata and fails loudly -- it cannot be
    silently wrong."""
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage as _PS
    from torchtitan.distributed import pipeline_parallel as PPmod

    def make_stage(submodule, stage_index, num_stages, device, *args, **kw):
        group = kw.get("group")
        fake = group is not None and dist.get_backend(group) == "fake"
        if fake and kw.get("input_args") is None:
            kw.update(_static_meta(stage_index, num_stages))
        return _PS(submodule, stage_index, num_stages, device, *args, **kw)

    PPmod.PipelineStage = make_stage
    if _ARGS.real:
        return

    # A fake recv never writes its buffer; allocate P2P buffers zero-filled so a
    # stage's input is finite (uninitialised memory can hold NaN/Inf, which is
    # harmless for dense-GEMM timing but would trip TorchTitan's loss check).
    from torch.distributed.pipelining import _utils as PU, stage as PSt
    orig_make = PU._make_tensor_from_meta

    def make_tensor(meta, device):
        t = orig_make(meta, device)
        return t.zero_() if t.is_floating_point() else t

    PU._make_tensor_from_meta = make_tensor
    PSt._make_tensor_from_meta = make_tensor

    # Belt and braces: never abort a fake run on a non-finite loss.
    import math as _math
    import types as _types
    import torchtitan.trainer as T
    T.math = _types.SimpleNamespace(**{k: getattr(_math, k) for k in dir(_math)
                                        if not k.startswith("__")})
    T.math.isfinite = lambda x: True


def _static_meta(stage_idx: int, num_stages: int) -> dict:
    import torch
    from torchtitan.config import TORCH_DTYPE_MAP

    cfg = _CONFIG
    par, tr = cfg.parallelism, cfg.training
    model = cfg.model_spec.model
    mbs = par.pipeline_parallel_microbatch_size
    cp = max(1, par.context_parallel_degree)
    tp = max(1, par.tensor_parallel_degree)
    seq = tr.seq_len // cp
    hidden_seq = seq
    dtype = TORCH_DTYPE_MAP[tr.mixed_precision_param]
    env_dtype = os.environ.get("DAGTRACE_PP_DTYPE")
    if env_dtype:
        dtype = getattr(torch, env_dtype)
    if os.environ.get("DAGTRACE_PP_SP_SHARDED", "1") == "1" and tp > 1 \
            and par.enable_sequence_parallel:
        hidden_seq = seq // tp

    def ex(*shape, dt=dtype):
        return torch.empty(shape, dtype=dt, device="meta")

    hidden = ex(mbs, hidden_seq, model.dim).requires_grad_()
    lm_head_in_loss = "Chunked" in type(cfg.loss).__name__
    if lm_head_in_loss:
        last_out = ex(mbs, hidden_seq, model.dim)
    else:
        vocab = model.vocab_size // tp if tp > 1 else model.vocab_size
        last_out = ex(mbs, seq, vocab)
    last_out.requires_grad_()
    first, last = stage_idx == 0, stage_idx == num_stages - 1
    hg = hidden.detach()
    return {
        "input_args": (ex(mbs, seq, dt=torch.int64) if first else hidden,),
        "output_args": (last_out if last else hidden,),
        "input_grads": (None,) if first else (hg,),
        "output_grads": (None,) if last else (hg,),
    }


# ------------------------------------------------------------- trainer hook
def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _model_scalars(obj, prefix="", depth=0, out=None):
    """Flatten scalar fields of a (nested) config dataclass for metadata."""
    out = {} if out is None else out
    if depth > 3 or obj is None:
        return out
    fields = getattr(obj, "__dataclass_fields__", None)
    if not fields:
        return out
    for k in fields:
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
        elif isinstance(v, (list, tuple)) and k == "layers":
            out[key + ".count"] = len(v)
            if v:
                _model_scalars(v[0], key + "[0].", depth + 1, out)
        elif hasattr(v, "__dataclass_fields__"):
            _model_scalars(v, key + ".", depth + 1, out)
    return out


def _dump_meta(trainer, path):
    import torch
    import torch.distributed as dist
    from dagtrace import topology

    pd = trainer.parallel_dims
    meshes = []
    for attr in ("_world_mesh", "_single_axis_meshes", "_multi_axis_meshes", "_global_meshes"):
        v = getattr(pd, attr, None)
        if isinstance(v, dict):
            meshes += list(v.values())
        elif v is not None:
            meshes.append(v)

    def local_numel(p):
        return p.to_local().numel() if hasattr(p, "to_local") else p.numel()

    parts = []
    for i, m in enumerate(trainer.model_parts):
        ps = list(m.parameters())
        parts.append({
            "index": i,
            "local_numel": int(sum(local_numel(p) for p in ps)),
            "global_numel": int(sum(p.numel() for p in ps)),
            "local_bytes": int(sum(local_numel(p) * p.element_size() for p in ps)),
            "param_dtypes": sorted({str(p.dtype) for p in ps}),
        })

    stages = []
    sch = getattr(trainer, "pp_schedule", None)
    if sch is not None:
        sts = getattr(sch, "_stages", None) or [getattr(sch, "_stage", None)]
        for s in sts:
            if s is not None:
                stages.append({"stage_index": s.stage_index, "num_stages": s.num_stages})

    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "capability": list(torch.cuda.get_device_capability(0)),
               "total_memory": p.total_memory, "sm_count": p.multi_processor_count,
               "driver_cuda": torch.version.cuda,
               "nccl": ".".join(map(str, torch.cuda.nccl.version())) if dist.is_nccl_available() else None}
    try:
        import torchtitan
        tt_ver = getattr(torchtitan, "__version__", "?")
        tt_path = os.path.dirname(torchtitan.__file__)
    except Exception:
        tt_ver, tt_path = "?", "?"
    try:
        cfg_dict = trainer.config.to_dict()
    except Exception:
        cfg_dict = None

    meta = {
        "tool": "dagtrace.titan_capture",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "mode": "real" if _ARGS.real else "fake",
        "argv": sys.argv,
        "capture_args": vars(_ARGS),
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "requested_pp_rank": None if _ARGS.real else (_ARGS.pp_rank if _ARGS.rank is None else None),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                     "torchtitan": tt_ver, "torchtitan_path": tt_path,
                     "python": platform.python_version()},
        "gpu": gpu,
        "parallel_dims": {k: getattr(pd, k, None) for k in
                          ("pp", "dp_replicate", "dp_shard", "cp", "tp", "ep", "etp", "world_size")},
        "model": {"name": trainer.config.model_spec.name,
                  "flavor": trainer.config.model_spec.flavor,
                  "fields": _model_scalars(trainer.config.model_spec.model)},
        "training": {"local_batch_size": trainer.config.training.local_batch_size,
                     "seq_len": trainer.config.training.seq_len,
                     "mixed_precision_param": trainer.config.training.mixed_precision_param,
                     "mixed_precision_reduce": getattr(trainer.config.training, "mixed_precision_reduce", None),
                     "pp_microbatch_size": trainer.config.parallelism.pipeline_parallel_microbatch_size,
                     "pp_schedule": trainer.config.parallelism.pipeline_parallel_schedule,
                     "sequence_parallel": trainer.config.parallelism.enable_sequence_parallel,
                     "loss": type(trainer.config.loss).__name__},
        "model_parts": parts,
        "pp_stages": stages,
        "topology": topology.collect(meshes),
        "config": cfg_dict,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def _install_direct_hook():
    import hashlib
    import torchtitan.trainer as T
    from dagtrace import online

    orig = T.Trainer.train_step
    state = {"meta": False}

    def train_step(self, data_iterator):
        a = _ARGS
        step = self.step
        if step <= a.warmup:
            return orig(self, data_iterator)
        import torch.distributed as dist
        rank = dist.get_rank()
        d = os.path.join(a.out, f"rank{rank}")
        os.makedirs(d, exist_ok=True)
        if not state["meta"]:
            meta = _dump_meta(self, os.path.join(d, "meta.json"))
            state["meta"] = True
            _check_rank(meta)
        _sync()
        online.TRACER.begin()
        if a.op_names:
            with online.OpCounter(online.TRACER):
                out = orig(self, data_iterator)
        else:
            out = orig(self, data_iterator)
        dag = online.TRACER.end(min_us=a.min_segment_us)
        struct = [[{k: v for k, v in n.items() if k in ("id", "type", "op", "bytes", "group", "peer")}
                   for n in dag["nodes"]], dag["edges"]]
        doc = {"format": "dagtrace.gpu_dag/1", "rank": rank, "world_size": dist.get_world_size(),
               "step": step, "mode": "real" if a.real else "fake",
               "signature": hashlib.sha256(json.dumps(struct).encode()).hexdigest(),
               "groups": {k: v for k, v in online.TRACER.groups.items()},
               "nodes": dag["nodes"], "edges": dag["edges"]}
        p = os.path.join(d, f"gpu_dag.step{step}.json")
        with open(p, "w") as f:
            json.dump(doc, f)
        n = collections.Counter(x["type"] for x in dag["nodes"])
        print(f"[dagtrace] rank {rank} step {step}: {dict(n)}, {len(dag['edges'])} edges, "
              f"compute {sum(x.get('dur_us', 0) for x in dag['nodes'])/1e3:.2f} ms, "
              f"sig {doc['signature'][:12]} -> {p}", flush=True)
        return out

    T.Trainer.train_step = train_step


def _install_trainer_hook():
    import torch
    from torch.profiler import ExecutionTraceObserver, ProfilerActivity, profile, record_function
    import torchtitan.trainer as T
    from dagtrace import commtag

    orig = T.Trainer.train_step
    state = {"meta": False}

    def train_step(self, data_iterator):
        a = _ARGS
        step = self.step
        if step <= a.warmup:
            return orig(self, data_iterator)
        import torch.distributed as dist
        rank = dist.get_rank()
        d = os.path.join(a.out, f"rank{rank}")
        os.makedirs(d, exist_ok=True)
        if not state["meta"]:
            meta = _dump_meta(self, os.path.join(d, "meta.json"))
            state["meta"] = True
            _check_rank(meta)
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        _sync()
        commtag.reset_pending()
        et_path = os.path.join(d, f"step{step}.et.json")
        et = ExecutionTraceObserver().register_callback(et_path)
        with profile(activities=acts, record_shapes=True, execution_trace_observer=et) as prof:
            commtag.calibrate_streams()
            with record_function(f"dagtrace::step#{step}"):
                out = orig(self, data_iterator)
                _sync()
        prof.export_chrome_trace(os.path.join(d, f"step{step}.kineto.json"))
        print(f"[dagtrace] rank {rank}: wrote {et_path} (+ kineto)", flush=True)
        return out

    T.Trainer.train_step = train_step


def _check_rank(meta):
    """Make sure the emulated rank sits at the requested PP coordinate."""
    if _ARGS.real or meta["requested_pp_rank"] is None:
        return
    pp = meta["parallel_dims"]["pp"] or 1
    if pp <= 1:
        return
    r = meta["rank"]
    for g in meta["topology"]["groups"].values():
        if "pp" in g["axes"] and g["partition"]:
            for blk in g["partition"]:
                if r in blk and blk.index(r) != meta["requested_pp_rank"]:
                    raise SystemExit(f"[dagtrace] rank {r} is pp coordinate {blk.index(r)}, "
                                     f"expected {meta['requested_pp_rank']}")


# --------------------------------------------------------------------- main
def _chdir_to_torchtitan_source():
    """TorchTitan configs reference ./tests/assets/... and ./assets/...; run from
    the TorchTitan source checkout (TORCHTITAN_DIR, else the installed package's
    parent directory if it is a source checkout)."""
    cand = os.environ.get("TORCHTITAN_DIR")
    if not cand:
        import importlib.util
        spec = importlib.util.find_spec("torchtitan")
        if spec and spec.origin:
            cand = os.path.dirname(os.path.dirname(spec.origin))
    if cand and os.path.isdir(os.path.join(cand, "tests", "assets")):
        os.chdir(cand)
    elif not os.path.isdir(os.path.join("tests", "assets")):
        print("[dagtrace] WARNING: TorchTitan source checkout not found; set TORCHTITAN_DIR "
              "(needed for the bundled test tokenizer/dataset)", flush=True)


def main():
    global _ARGS
    _ARGS, tt_args = _parse()
    _ARGS.out = os.path.abspath(_ARGS.out)
    os.makedirs(_ARGS.out, exist_ok=True)
    _chdir_to_torchtitan_source()
    if not _ARGS.real:
        os.environ["NGPU"] = str(_ARGS.ngpu)
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
    if _ARGS.cpu_test:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        import cpu_shim  # noqa: F401
    _install_config_hook()
    _install_fake_pp_patch()
    if _ARGS.mode == "direct":
        from dagtrace import online
        online.install()
        _install_direct_hook()
    else:
        from dagtrace import commtag
        commtag.install()
        _install_trainer_hook()
    from torchtitan.train import main as tt_main
    if not _ARGS.keep_cuda_graphs and not any("disable_cuda_graphs" in a for a in tt_args):
        # must be on the command line: TorchTitan validates it while parsing
        tt_args = tt_args + ["--training.disable_cuda_graphs"]
    sys.argv = ["torchtitan.train"] + tt_args
    tt_main()


if __name__ == "__main__":
    main()
