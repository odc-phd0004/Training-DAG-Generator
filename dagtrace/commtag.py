"""Tag every torch.distributed collective / P2P call so that it can be recovered
from a PyTorch Execution Trace (ET) together with its process group and peer.

Why this exists
---------------
With the ``fake`` process-group backend no NCCL kernel is launched and the
``record_param_comms`` annotations that the stock Chakra converter keys on are
never emitted.  The dispatcher-level ``c10d::*`` ops *are* recorded in the ET
(with tensor shapes and dtypes), but their ProcessGroup argument is opaque
(``"<Object>"``).  Functional collectives (``_c10d_functional::*``, used by
DTensor/TP) carry their group name explicitly, so they need nothing extra.

This module wraps the public ``torch.distributed`` collective APIs with a
``record_function`` whose name encodes::

    dagcomm::<api>#<seq>|pg=<group_name>|peer=<global peer rank or ->|async=<0/1>

and wraps ``Work.wait`` (and subclasses such as ``FakeWork``) with
``dagwait::<seq>`` so that the point where the compute stream starts waiting
for an async collective is also visible in the trace.

The wrapping is purely observational: arguments and return values are passed
through untouched, so the training program's behaviour is unchanged.  It works
identically with the real NCCL backend, which is what makes the fake-vs-real
structural comparison (dagtrace.compare) meaningful.
"""
from __future__ import annotations

import functools
import inspect
import threading

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.autograd.profiler import record_function

COMM_TAG = "dagcomm::"
WAIT_TAG = "dagwait::"
EVT_TAG = "dagevt::"
CALIB_TAG = "dagcalib::"

# Collective / P2P entry points we tag. Object collectives (pickle-based) are
# intentionally excluded: they are control-plane traffic, not training traffic.
_APIS = [
    "all_reduce", "all_reduce_coalesced",
    "all_gather", "all_gather_into_tensor", "all_gather_single", "all_gather_coalesced",
    "_all_gather_base",
    "reduce_scatter", "reduce_scatter_tensor", "reduce_scatter_single", "_reduce_scatter_base",
    "all_to_all", "all_to_all_single",
    "broadcast", "reduce", "gather", "gather_into_tensor", "gather_single", "scatter",
    "send", "recv", "isend", "irecv",
    "barrier", "monitored_barrier",
]
_PEER_ARGS = {
    "send": ("dst", "group_dst"), "isend": ("dst", "group_dst"),
    "recv": ("src", "group_src"), "irecv": ("src", "group_src"),
    "broadcast": ("src", "group_src"), "scatter": ("src", "group_src"),
    "reduce": ("dst", "group_dst"), "gather": ("dst", "group_dst"),
    "gather_into_tensor": ("dst", "group_dst"), "gather_single": ("dst", "group_dst"),
}
_P2P = {"send", "recv", "isend", "irecv"}

_state = threading.local()
_lock = threading.Lock()
_seq = [0]
_pending: dict[int, tuple[int, object]] = {}   # id(work) -> (seq, work)
_installed = False


_stream_cache: dict[int, str] = {}
_libcuda = None


def _cu_stream_id(ptr: int):
    """Driver-level unique stream id (what CUPTI/Kineto reports as 'stream')."""
    global _libcuda
    import ctypes
    if _libcuda is None:
        try:
            _libcuda = ctypes.CDLL("libcuda.so.1")
            _libcuda.cuStreamGetId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        except Exception:
            _libcuda = False
    if not _libcuda:
        return None
    out = ctypes.c_ulonglong(0)
    rc = _libcuda.cuStreamGetId(ctypes.c_void_p(ptr), ctypes.byref(out))
    return int(out.value) if rc == 0 else None


def stream_key(stream=None) -> str:
    """Stable key for a CUDA stream: its driver stream id, else 'p<ptr>'.
    Returns 'main' when CUDA is unavailable (CPU test runs)."""
    if not torch.cuda.is_available():
        return "main"
    try:
        s = stream if stream is not None else torch.cuda.current_stream()
        ptr = int(s.cuda_stream)
    except Exception:
        return "?"
    k = _stream_cache.get(ptr)
    if k is None:
        sid = _cu_stream_id(ptr)
        k = str(sid) if sid is not None else f"p{ptr}"
        _stream_cache[ptr] = k
    return k


def _depth() -> int:
    return getattr(_state, "depth", 0)


def _group_name(group) -> str:
    try:
        pg = group if group is not None else c10d._get_default_group()
        return str(pg.group_name)
    except Exception:
        return "?"


def _global_peer(api: str, bound: inspect.BoundArguments, group) -> str:
    names = _PEER_ARGS.get(api)
    if not names:
        return "-"
    glob_name, grp_name = names
    a = bound.arguments
    if a.get(glob_name) is not None:
        return str(int(a[glob_name]))
    if a.get(grp_name) is not None:
        try:
            pg = group if group is not None else c10d._get_default_group()
            return str(dist.get_global_rank(pg, int(a[grp_name])))
        except Exception:
            return "g" + str(int(a[grp_name]))
    return "-"


def _wrap(api: str, fn):
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _depth() > 0:                      # nested call from another tagged API
            return fn(*args, **kwargs)
        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)
        group = bound.arguments.get("group")
        is_async = bool(bound.arguments.get("async_op", False)) or api in ("isend", "irecv")
        with _lock:
            _seq[0] += 1
            seq = _seq[0]
        tag = (f"{COMM_TAG}{api}#{seq}|pg={_group_name(group)}"
               f"|peer={_global_peer(api, bound, group)}|async={int(is_async)}"
               f"|stream={stream_key()}")
        _state.depth = _depth() + 1
        try:
            with record_function(tag):
                out = fn(*args, **kwargs)
        finally:
            _state.depth -= 1
        if is_async and out is not None and hasattr(out, "wait"):
            _pending[id(out)] = (seq, out)
        return out

    wrapper.__dagtrace_wrapped__ = True
    return wrapper


def _wrap_wait(cls):
    orig = cls.__dict__["wait"]

    def wait(self, *args, **kwargs):
        ent = _pending.pop(id(self), None)
        if ent is None:
            return orig(self, *args, **kwargs)
        with record_function(f"{WAIT_TAG}{ent[0]}|stream={stream_key()}"):
            return orig(self, *args, **kwargs)

    cls.wait = wait


def _all_subclasses(cls):
    out, stack = [], [cls]
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(c.__subclasses__())
    return out


_eid = [0]


def _event_id(ev) -> int:
    e = getattr(ev, "_dagtrace_eid", None)
    if e is None:
        _eid[0] += 1
        e = _eid[0]
        try:
            ev._dagtrace_eid = e
        except Exception:
            return id(ev)
    return e


def _install_event_hooks():
    """Record cross-stream synchronisation (FSDP2 overlaps comm with compute
    through side streams + events, not through async_op)."""
    if not torch.cuda.is_available():
        return
    Ev = torch.cuda.Event
    if getattr(Ev.record, "__dagtrace_wrapped__", False):
        return
    orig_rec, orig_wait = Ev.record, Ev.wait

    def record(self, stream=None):
        with record_function(f"{EVT_TAG}rec|e={_event_id(self)}|s={stream_key(stream)}"):
            return orig_rec(self, stream)

    def wait(self, stream=None):
        with record_function(f"{EVT_TAG}wait|e={_event_id(self)}|s={stream_key(stream)}"):
            return orig_wait(self, stream)

    record.__dagtrace_wrapped__ = wait.__dagtrace_wrapped__ = True
    Ev.record, Ev.wait = record, wait


def calibrate_streams() -> None:
    """Launch one tiny kernel on the current stream and on a fresh side stream,
    each inside a dagcalib::<key> region.  The extractor compares <key> with
    the stream id Kineto reports for that kernel, verifying the key mapping."""
    if not torch.cuda.is_available():
        return
    x = torch.zeros(1, device="cuda")
    for s in (torch.cuda.current_stream(), torch.cuda.Stream()):
        with torch.cuda.stream(s):
            with record_function(f"{CALIB_TAG}{stream_key(s)}"):
                x.add_(1)
    torch.cuda.synchronize()


def install() -> None:
    """Idempotently wrap torch.distributed APIs. Call before the model is built."""
    global _installed
    if _installed:
        return
    for api in _APIS:
        fn = getattr(c10d, api, None)
        if fn is None or getattr(fn, "__dagtrace_wrapped__", False):
            continue
        w = _wrap(api, fn)
        setattr(c10d, api, w)
        if getattr(dist, api, None) is fn:
            setattr(dist, api, w)
    # P2POp validates `op in (isend, irecv)` against the module globals we just
    # replaced, so batch_isend_irecv keeps working and its inner isend/irecv
    # calls get tagged individually (with their peers).
    for cls in _all_subclasses(c10d.Work):
        if "wait" in cls.__dict__:
            try:
                _wrap_wait(cls)
            except (TypeError, AttributeError):
                pass
    _install_event_hooks()
    _installed = True


def reset_pending() -> None:
    _pending.clear()


def parse_tag(name: str) -> dict | None:
    """Parse a dagcomm:: record_function name back into a dict."""
    if not name.startswith(COMM_TAG):
        return None
    head, *fields = name[len(COMM_TAG):].split("|")
    api, seq = head.split("#")
    d = {"api": api, "seq": int(seq)}
    for f in fields:
        k, v = f.split("=", 1)
        d[k] = v
    d["async"] = d.get("async") == "1"
    d.setdefault("stream", None)
    if d.get("peer") in ("-", None):
        d["peer"] = None
    elif not d["peer"].startswith("g"):
        d["peer"] = int(d["peer"])
    return d


def parse_kv(name: str, prefix: str) -> dict:
    """Parse 'prefix<head>|k=v|k=v' -> {'head': head, k: v}."""
    head, *fields = name[len(prefix):].split("|")
    d = {"head": head}
    for f in fields:
        if "=" in f:
            k, v = f.split("=", 1)
            d[k] = v
    return d
