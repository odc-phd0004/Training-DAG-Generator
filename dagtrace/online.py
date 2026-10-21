"""Direct DAG capture: build the compute/collective/P2P DAG *while the step
runs*, with no PyTorch profiler, no Execution Trace and no Kineto.

Vertices
  compute     GPU work on one CUDA stream between two dependency boundaries;
              dur_us measured with CUDA events recorded on that stream
  collective  one collective call (all_reduce, all_gather, reduce_scatter,
              all_to_all, broadcast, ...): op, bytes, process group
  p2p         one send or recv: op, bytes, peer rank

Edges = dependencies only, derived from CUDA/NCCL stream semantics:
  * compute on a stream follows the previous work on that stream
  * a collective/p2p starts after the work issued before it on its stream
  * collectives on one communicator run in order (p2p: per peer, per direction)
  * the stream resumes after a synchronous collective, after Work.wait() for
    async ones, and after wait_tensor() for functional (DTensor) collectives
  * cross-stream waits via torch.cuda.Event record/wait (how FSDP2 overlaps)

How it works: every collective / P2P entry point of torch.distributed and of
the functional-collective API (used by DTensor/TP), Work.wait, wait_tensor and
torch.cuda.Event.record/wait are wrapped.  Each such call is a "boundary": the
compute segment open on that stream is closed (CUDA event), the vertex and its
edges are added, and a new segment is opened (CUDA event).  Segments with no
measurable GPU work are removed at the end of the step, preserving
reachability.  Arguments and return values pass through untouched.

Timing note: a segment's time is the span between its two CUDA events on its
stream, so it includes GPU idle gaps *inside* the segment caused by slow CPU
kernel launches.  For realistic model sizes the GPU is the bottleneck and this
is negligible; for tiny models it inflates compute.  (The profiler-based path,
`titan_capture --raw` + `dagtrace.extract`, measures pure kernel time.)
"""
from __future__ import annotations

import collections
import functools
import inspect
import threading
import time

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

# python API -> (collective op name, output arg, input arg)
_C10D = {
    "all_reduce": ("all_reduce", "tensor", "tensor"),
    "all_reduce_coalesced": ("all_reduce", "tensors", "tensors"),
    "all_gather": ("all_gather", "tensor_list", "tensor"),
    "all_gather_into_tensor": ("all_gather", "output_tensor", "input_tensor"),
    "all_gather_single": ("all_gather", "output_tensor", "input_tensor"),
    "all_gather_coalesced": ("all_gather", "output_tensor_lists", "input_tensor_list"),
    "_all_gather_base": ("all_gather", "output_tensor", "input_tensor"),
    "reduce_scatter": ("reduce_scatter", "output", "input_list"),
    "reduce_scatter_tensor": ("reduce_scatter", "output", "input"),
    "reduce_scatter_single": ("reduce_scatter", "output", "input"),
    "_reduce_scatter_base": ("reduce_scatter", "output", "input"),
    "all_to_all": ("all_to_all", "output_tensor_list", "input_tensor_list"),
    "all_to_all_single": ("all_to_all", "output", "input"),
    "broadcast": ("broadcast", "tensor", "tensor"),
    "reduce": ("reduce", "tensor", "tensor"),
    "gather": ("gather", "gather_list", "tensor"),
    "gather_into_tensor": ("gather", "gather_tensor", "tensor"),
    "gather_single": ("gather", "gather_tensor", "tensor"),
    "scatter": ("scatter", "tensor", "scatter_list"),
    "barrier": ("barrier", None, None),
    "send": ("send", None, "tensor"), "isend": ("send", None, "tensor"),
    "recv": ("recv", "tensor", None), "irecv": ("recv", "tensor", None),
}
_PEER = {"send": ("dst", "group_dst"), "isend": ("dst", "group_dst"),
         "recv": ("src", "group_src"), "irecv": ("src", "group_src")}
# functional collectives (DTensor / tensor parallel): name -> (op, size rule)
_FUNCOL = {
    "all_reduce": "all_reduce", "all_reduce_coalesced": "all_reduce",
    "all_gather_tensor": "all_gather", "all_gather_tensor_autograd": "all_gather",
    "all_gather_single": "all_gather", "all_gather_single_autograd": "all_gather",
    "all_gather_into_tensor_coalesced": "all_gather", "all_gather_single_coalesced": "all_gather",
    "reduce_scatter_tensor": "reduce_scatter", "reduce_scatter_tensor_autograd": "reduce_scatter",
    "reduce_scatter_single": "reduce_scatter", "reduce_scatter_single_autograd": "reduce_scatter",
    "reduce_scatter_tensor_coalesced": "reduce_scatter",
    "reduce_scatter_single_coalesced": "reduce_scatter",
    "all_to_all_single": "all_to_all", "all_to_all_single_autograd": "all_to_all",
    "broadcast": "broadcast",
}


def _nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(t) for t in x)
    return 0


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for t in x:
            r = _first_tensor(t)
            if r is not None:
                return r
    return None


class Tracer:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.active = False
        self.tls = threading.local()
        self._reset()

    # ------------------------------------------------------------ state
    def _reset(self):
        self.nodes = []                      # dicts
        self.deps = collections.defaultdict(set)
        self.F = collections.defaultdict(set)    # stream -> frontier
        self.open = {}                       # stream -> segment id
        self.timers = {}                     # segment id -> (start, end)
        self.events = {}                     # event id -> frontier snapshot
        self.fifo = {}
        self.pending_work = {}               # id(work) -> (work, comm id)
        self.pending_tensor = {}             # data_ptr -> comm id
        self.groups = {}
        self.ops = collections.defaultdict(collections.Counter)   # segment -> op counts

    def phase(self):
        if getattr(self, "in_opt", False):
            return "opt"
        if getattr(self, "after_opt", False):
            return "post"
        return "bwd" if torch._C._current_graph_task_id() != -1 else "fwd"

    def _internal(self):
        return getattr(self.tls, "internal", False)

    def _stream(self, s=None):
        if s is not None and hasattr(s, "cuda_stream"):
            return int(s.cuda_stream)
        if not self.cuda:
            return "main"
        return int(torch.cuda.current_stream().cuda_stream)

    def _stamp(self, stream_key, stream_obj=None):
        if not self.cuda:
            return time.perf_counter()
        e = torch.cuda.Event(enable_timing=True)
        self.tls.internal = True
        try:
            if stream_obj is None:
                e.record()
            else:
                e.record(stream_obj)
        finally:
            self.tls.internal = False
        return e

    # ------------------------------------------------------ segments
    def _add(self, node, deps):
        node["id"] = len(self.nodes)
        self.nodes.append(node)
        self.deps[node["id"]] |= set(deps)
        return node["id"]

    def _open(self, s, stream_obj=None):
        sid = self._add({"type": "compute"}, self.F[s])
        self.F[s] = {sid}
        self.open[s] = sid
        self.timers[sid] = [self._stamp(s, stream_obj), None, s]
        self.nodes[sid]["_ph0"] = self.phase()
        return sid

    def _cut(self, s, stream_obj=None):
        sid = self.open.pop(s, None)
        if sid is not None:
            self.timers[sid][1] = self._stamp(s, stream_obj)
            self.nodes[sid]["_ph1"] = self.phase()

    def _reopen(self, s, stream_obj=None):
        self._open(s, stream_obj)

    # ------------------------------------------------------- boundaries
    def comm(self, op, nbytes, group_ranks, group_name, peer, is_async, wait_key=None,
             stream_obj=None):
        s = self._stream(stream_obj)
        self._cut(s, stream_obj)
        kind = "p2p" if op in ("send", "recv") else "collective"
        node = {"type": kind, "op": op, "bytes": int(nbytes), "group": group_name,
                "phase": self.phase()}
        if kind == "p2p":
            node["peer"] = peer
            q = ("p2p", group_name, peer, op)
        else:
            q = ("coll", group_name)
        deps = set(self.F[s])
        if q in self.fifo:
            deps.add(self.fifo[q])
        cid = self._add(node, deps)
        self.fifo[q] = cid
        if group_name is not None and group_name not in self.groups:
            self.groups[group_name] = group_ranks
        if not is_async:
            self.F[s] = {cid}
        elif wait_key is not None:
            self.pending_tensor[wait_key] = cid
        self._reopen(s, stream_obj)
        return cid

    def wait_comm(self, cid):
        s = self._stream()
        self._cut(s)
        self.F[s].add(cid)
        self._reopen(s)

    def event_record(self, eid, stream_obj):
        s = self._stream(stream_obj)
        self._cut(s, stream_obj)
        self.events[eid] = set(self.F[s])
        self._reopen(s, stream_obj)

    def event_wait(self, eid, stream_obj):
        s = self._stream(stream_obj)
        self._cut(s, stream_obj)
        self.F[s] |= self.events.get(eid, set())
        self._reopen(s, stream_obj)

    # -------------------------------------------------------------- step
    def begin(self):
        self._reset()
        self.in_opt = self.after_opt = False
        self.active = True
        self._t0 = self._stamp(self._stream())
        self._open(self._stream())

    def end(self, min_us=2.0):
        """Close the step and return the per-rank DAG (pruned)."""
        for s in list(self.open):
            self._cut(s)
        self.active = False
        if self.cuda:
            torch.cuda.synchronize()
        for sid, (a, b, s) in self.timers.items():
            if b is None:
                d = 0.0
            elif self.cuda:
                d = a.elapsed_time(b) * 1e3
            else:
                d = (b - a) * 1e6
            n = self.nodes[sid]
            n["dur_us"] = round(d, 3)
            n["_stream"] = s
            a0, a1 = n.get("_ph0", "fwd"), n.get("_ph1", n.get("_ph0", "fwd"))
            n["phase"] = a0 if a0 == a1 else f"{a0}->{a1}"
            if self.ops.get(sid):
                n["ops"] = dict(self.ops[sid].most_common(5))
        return self._prune(min_us)

    def _prune(self, min_us):
        """Remove empty compute segments, preserving reachability; renumber."""
        succ = collections.defaultdict(set)
        for v, ds in self.deps.items():
            for d in ds:
                succ[d].add(v)
        pred = {v: set(ds) for v, ds in self.deps.items()}
        keep = []
        for n in self.nodes:          # creation order is a topological order
            v = n["id"]
            if n["type"] == "compute" and n.get("dur_us", 0) < min_us:
                ps, ss = pred.get(v, set()), succ.get(v, set())
                for q in ss:
                    pred[q].discard(v)
                    pred[q] |= ps
                for p in ps:
                    succ[p].discard(v)
                    succ[p] |= ss
                pred.pop(v, None)
                succ.pop(v, None)
            else:
                keep.append(n)
        new = {n["id"]: i for i, n in enumerate(keep)}
        out_nodes, edges = [], set()
        for n in keep:
            m = {k: v for k, v in n.items() if not k.startswith("_")}
            m["id"] = new[n["id"]]
            out_nodes.append(m)
            for p in pred.get(n["id"], ()):
                if p in new:
                    edges.add((new[p], new[n["id"]]))
        # drop edges implied by a same-stream chain? keep: they are true deps
        return {"nodes": out_nodes, "edges": sorted(edges)}


TRACER = Tracer()


class OpCounter:
    """Optional (--op-names): count ATen ops per compute segment with a
    TorchDispatchMode.  Adds Python overhead per op, so it is off by default."""

    def __init__(self, tracer):
        from torch.utils._python_dispatch import TorchDispatchMode

        T = tracer

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if T.active:
                    sid = T.open.get(T._stream())
                    if sid is not None:
                        name = func.__name__.split(".")[0]
                        if name not in _IGNORED_OPS:
                            T.ops[sid][name] += 1
                return func(*args, **(kwargs or {}))

        self.mode = _Mode()

    def __enter__(self):
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self.mode.__exit__(*exc)


# metadata-only ops that launch no kernels
_IGNORED_OPS = {"view", "_unsafe_view", "reshape", "t", "transpose", "permute", "expand",
                "detach", "alias", "as_strided", "slice", "select", "split", "split_with_sizes",
                "unsqueeze", "squeeze", "empty", "empty_like", "empty_strided", "new_empty",
                "new_empty_strided", "chunk", "narrow", "unbind", "_to_copy_meta",
                "is_same_size", "lift_fresh", "_has_compatible_shallow_copy_type", "size",
                "stride", "sym_size", "sym_stride", "numel", "dim", "unflatten", "flatten",
                "_record_function_enter_new", "_record_function_exit", "_record_function_enter",
                "record_stream", "set_", "resize_", "_local_scalar_dense"}


# ------------------------------------------------------------------ hooks
def _group_info(group):
    pg = group if group is not None else c10d._get_default_group()
    try:
        return str(pg.group_name), dist.get_process_group_ranks(pg)
    except Exception:
        return None, None


def _wrap_c10d(api, fn):
    sig = inspect.signature(fn)
    op, out_arg, in_arg = _C10D[api]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        T = TRACER
        if not T.active or getattr(T.tls, "depth", 0) > 0:
            return fn(*args, **kwargs)
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
        except TypeError:
            return fn(*args, **kwargs)
        group = b.get("group")
        gname, granks = _group_info(group)
        if op == "all_gather":
            nb = _nbytes(b.get(out_arg))
        elif op in ("reduce_scatter", "scatter"):
            nb = _nbytes(b.get(in_arg))
        elif op == "gather":
            nb = _nbytes(b.get(out_arg)) or _nbytes(b.get(in_arg))
        else:
            nb = _nbytes(b.get(in_arg if in_arg else out_arg))
        peer = None
        if api in _PEER:
            g, gl = _PEER[api]
            if b.get(g) is not None:
                peer = int(b[g])
            elif b.get(gl) is not None and granks:
                peer = granks[int(b[gl])]
        is_async = bool(b.get("async_op", False)) or api in ("isend", "irecv")
        # close the compute segment BEFORE the call: kernels launched inside the
        # collective call (fake-backend local copies) must not count as compute
        T._cut(T._stream())
        T.tls.depth = getattr(T.tls, "depth", 0) + 1
        try:
            out = fn(*args, **kwargs)
        finally:
            T.tls.depth -= 1
        cid = T.comm(op, nb, granks, gname, peer, is_async)
        if is_async and out is not None and hasattr(out, "wait"):
            T.pending_work[id(out)] = (out, cid)
        return out

    wrapper.__dagtrace_online__ = True
    return wrapper


def _wrap_funcol(name, fn, funcol):
    op = _FUNCOL[name]
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        T = TRACER
        if not T.active or getattr(T.tls, "depth", 0) > 0:
            return fn(*args, **kwargs)
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
        except TypeError:
            return fn(*args, **kwargs)
        group = b.get("group")
        try:
            gname = str(funcol._resolve_group_name(group, b.get("tag", "")))
            granks = dist.get_process_group_ranks(c10d._resolve_process_group(gname))
        except Exception:
            gname, granks = None, None
        x = b.get("self", b.get("input", b.get("inputs", b.get("self_tensors"))))
        nb = _nbytes(x)
        if op == "all_gather" and granks:
            nb *= len(granks)
        # close the compute segment BEFORE the call: kernels launched inside the
        # collective call (fake-backend local copies) must not count as compute
        T._cut(T._stream())
        T.tls.depth = getattr(T.tls, "depth", 0) + 1
        try:
            out = fn(*args, **kwargs)
        finally:
            T.tls.depth -= 1
        t = _first_tensor(out)
        elem = getattr(t, "elem", t) if t is not None else None
        key = elem.data_ptr() if isinstance(elem, torch.Tensor) else None
        # if the result is not an AsyncCollectiveTensor it was already waited
        is_async = t is not None and type(t).__name__ == "AsyncCollectiveTensor"
        T.comm(op, nb, granks, gname, None, is_async, wait_key=key)
        return out

    wrapper.__dagtrace_online__ = True
    return wrapper


def _wrap_wait_tensor(fn):
    @functools.wraps(fn)
    def wait_tensor(tensor):
        T = TRACER
        out = fn(tensor)
        if T.active and isinstance(tensor, torch.Tensor):
            cid = T.pending_tensor.pop(tensor.data_ptr(), None)
            if cid is not None:
                T.wait_comm(cid)
        return out

    wait_tensor.__dagtrace_online__ = True
    return wait_tensor


def _wrap_work_wait(cls):
    orig = cls.__dict__["wait"]

    def wait(self, *a, **k):
        out = orig(self, *a, **k)
        T = TRACER
        if T.active:
            ent = T.pending_work.pop(id(self), None)
            if ent is not None:
                T.wait_comm(ent[1])
        return out

    cls.wait = wait


_eid = [0]


def _event_id(ev):
    e = getattr(ev, "_dagtrace_eid", None)
    if e is None:
        _eid[0] += 1
        e = _eid[0]
        try:
            ev._dagtrace_eid = e
        except Exception:
            return id(ev)
    return e


_installed = False


def install():
    global _installed
    if _installed:
        return
    for api in _C10D:
        fn = getattr(c10d, api, None)
        if fn is None or getattr(fn, "__dagtrace_online__", False):
            continue
        w = _wrap_c10d(api, fn)
        setattr(c10d, api, w)
        if getattr(dist, api, None) is fn:
            setattr(dist, api, w)
    import torch.distributed._functional_collectives as funcol
    for name in _FUNCOL:
        fn = getattr(funcol, name, None)
        if fn is not None and not getattr(fn, "__dagtrace_online__", False):
            setattr(funcol, name, _wrap_funcol(name, fn, funcol))
    funcol.wait_tensor = _wrap_wait_tensor(funcol.wait_tensor)
    stack = [c10d.Work]
    while stack:
        c = stack.pop()
        stack.extend(c.__subclasses__())
        if "wait" in c.__dict__:
            try:
                _wrap_work_wait(c)
            except (TypeError, AttributeError):
                pass
    import torch.optim.optimizer as O

    def _phase_change(new):
        # a phase change is a segment boundary (no dependency change)
        T = TRACER
        s = T._stream() if T.active else None
        reopen = s is not None and s in T.open
        if reopen:
            T._cut(s)
        T.in_opt, T.after_opt = new == "opt", new == "post"
        if reopen:
            T._reopen(s)

    def _pre(opt, args, kwargs):
        _phase_change("opt")

    def _post(opt, args, kwargs):
        _phase_change("post")

    O.register_optimizer_step_pre_hook(_pre)
    O.register_optimizer_step_post_hook(_post)
    if torch.cuda.is_available():
        Ev = torch.cuda.Event
        orig_rec, orig_wait = Ev.record, Ev.wait

        def record(self, stream=None):
            out = orig_rec(self, stream)
            T = TRACER
            if T.active and not T._internal():
                T.event_record(_event_id(self), stream)
            return out

        def wait(self, stream=None):
            out = orig_wait(self, stream)
            T = TRACER
            if T.active and not T._internal():
                T.event_wait(_event_id(self), stream)
            return out

        Ev.record, Ev.wait = record, wait
    _installed = True
