"""Turn one rank's PyTorch Execution Trace (+ Kineto trace) into a
compute/communication DAG.

Dependency model (mirrors CUDA stream semantics used by NCCL process groups):

  * compute units execute in issue order on the compute stream     (edge "seq")
  * a collective starts after the compute issued just before it    (edge "issue")
    -- ProcessGroupNCCL makes its stream wait on the current stream at issue
  * collectives on the same communicator run in FIFO order         (edge "pg_fifo")
    (P2P: per (group, peer) pair)
  * the compute stream waits for a collective at its wait point:   (edge "complete")
      - synchronous c10d call      -> right after the call
      - async c10d call            -> Work.wait()   (dagwait:: tag)
      - functional collective      -> _c10d_functional::wait_tensor on its output
      - otherwise                  -> first op that reads the collective's output
    and the first compute unit issued after that point depends on it.

A "compute unit" is an outermost dispatcher op (aten::mm, aten::_flash_..., a
fused optimizer op, ...) or an annotated region that launches GPU kernels with
no dispatcher op around it (e.g. a torch.compile'd Triton region).  Its time
is the sum of its GPU kernel durations from Kineto (device time).  If the
trace has no GPU kernels at all (CPU test runs) CPU op time is used instead.

Outputs (see write_outputs):
  rank<R>.step<k>.fine.json.gz   every compute unit / comm op
  rank<R>.step<k>.dag.json       coarse DAG: maximal compute segments between
                                 communication boundaries + comm nodes
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re

try:
    import orjson as _fastjson
except Exception:  # pragma: no cover
    _fastjson = None

from dagtrace.commtag import CALIB_TAG, COMM_TAG, EVT_TAG, WAIT_TAG, parse_kv, parse_tag

ROOT_PREFIX = "[pytorch|profiler|execution_trace"

# ------------------------------------------------------------------ helpers

def load_json(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        data = f.read()
    return _fastjson.loads(data) if _fastjson else json.loads(data)


def _attr(node, name, default=None):
    for a in node.get("attrs", ()):
        if a["name"] == name:
            return a["value"]
    return default


def _is_tensor_val(v):
    return (isinstance(v, list) and len(v) == 6 and isinstance(v[0], int)
            and isinstance(v[3], int) and isinstance(v[4], int))


def _tensors(v):
    """Flatten an ET value into a list of (tensor_id, storage_id, numel, itemsize)."""
    if _is_tensor_val(v):
        return [(v[0], v[1], v[3], v[4])]
    out = []
    if isinstance(v, list):
        for x in v:
            out += _tensors(x)
    return out


def _dtype_of(typ):
    m = re.search(r"Tensor\(([^)]*)\)", typ or "")
    return m.group(1) if m else None


def _nbytes(ts):
    return sum(n * s for _, _, n, s in ts)


# --------------------------------------------------------- collective tables
# op name -> (collective type, index of output arg, index of input arg)
# index None means "same as the other" (in-place), "outs" means use ET outputs.
C10D_OPS = {
    "c10d::_allgather_base_": ("ALL_GATHER", 0, 1),
    "c10d::allgather_": ("ALL_GATHER", 0, 1),
    "c10d::allgather_into_tensor_coalesced_": ("ALL_GATHER", 0, 1),
    "c10d::allgather_coalesced_": ("ALL_GATHER", 0, 1),
    "c10d::_reduce_scatter_base_": ("REDUCE_SCATTER", 0, 1),
    "c10d::reduce_scatter_": ("REDUCE_SCATTER", 0, 1),
    "c10d::reduce_scatter_tensor_coalesced_": ("REDUCE_SCATTER", 0, 1),
    "c10d::allreduce_": ("ALL_REDUCE", 0, 0),
    "c10d::allreduce_coalesced_": ("ALL_REDUCE", 0, 0),
    "c10d::alltoall_base_": ("ALL_TO_ALL", 0, 1),
    "c10d::alltoall_": ("ALL_TO_ALL", 0, 1),
    "c10d::broadcast_": ("BROADCAST", 0, 0),
    "c10d::reduce_": ("REDUCE", 0, 0),
    "c10d::gather_": ("GATHER", 0, 1),
    "c10d::scatter_": ("SCATTER", 0, 1),
    "c10d::send": ("SEND", None, 0),
    "c10d::recv_": ("RECV", 0, None),
    "c10d::recv_any_source_": ("RECV", 0, None),
    "c10d::barrier": ("BARRIER", None, None),
    "c10d::monitored_barrier_": ("BARRIER", None, None),
}
FUNC_OPS = {
    "all_reduce": "ALL_REDUCE", "all_reduce_": "ALL_REDUCE", "all_reduce_coalesced": "ALL_REDUCE",
    "all_reduce_coalesced_": "ALL_REDUCE",
    "all_gather_into_tensor": "ALL_GATHER", "all_gather_into_tensor_out": "ALL_GATHER",
    "all_gather_into_tensor_coalesced": "ALL_GATHER",
    "reduce_scatter_tensor": "REDUCE_SCATTER", "reduce_scatter_tensor_out": "REDUCE_SCATTER",
    "reduce_scatter_tensor_coalesced": "REDUCE_SCATTER",
    "all_to_all_single": "ALL_TO_ALL",
    "broadcast": "BROADCAST", "broadcast_": "BROADCAST",
}
FUNC_NS = ("_c10d_functional::", "_c10d_functional_autograd::")
WAIT_OPS = {"_c10d_functional::wait_tensor", "_c10d_functional_autograd::wait_tensor"}

# which c10d op types a tagged python API may produce (for tag <-> op matching)
_API_TO_COLL = {
    "all_reduce": {"ALL_REDUCE"}, "all_reduce_coalesced": {"ALL_REDUCE"},
    "all_gather": {"ALL_GATHER"}, "all_gather_into_tensor": {"ALL_GATHER"},
    "all_gather_single": {"ALL_GATHER"}, "all_gather_coalesced": {"ALL_GATHER"},
    "_all_gather_base": {"ALL_GATHER"},
    "reduce_scatter": {"REDUCE_SCATTER"}, "reduce_scatter_tensor": {"REDUCE_SCATTER"},
    "reduce_scatter_single": {"REDUCE_SCATTER"}, "_reduce_scatter_base": {"REDUCE_SCATTER"},
    "all_to_all": {"ALL_TO_ALL"}, "all_to_all_single": {"ALL_TO_ALL"},
    "broadcast": {"BROADCAST"}, "reduce": {"REDUCE"},
    "gather": {"GATHER", "ALL_GATHER"}, "gather_into_tensor": {"GATHER", "ALL_GATHER"},
    "gather_single": {"GATHER", "ALL_GATHER"}, "scatter": {"SCATTER"},
    "send": {"SEND"}, "isend": {"SEND"}, "recv": {"RECV"}, "irecv": {"RECV"},
    "barrier": {"BARRIER", "ALL_REDUCE"}, "monitored_barrier": {"BARRIER"},
}


def nccl_size(coll, in_b, out_b):
    """Message size with nccl-tests conventions."""
    if coll == "ALL_GATHER":
        return out_b
    if coll in ("REDUCE_SCATTER", "SCATTER"):
        return in_b
    if coll == "GATHER":
        return out_b or in_b
    return max(in_b, out_b)


# ------------------------------------------------------------- the extractor

class RankTrace:
    def __init__(self, et_path, kineto_path=None, meta=None, mode=None):
        self.meta = meta or {}
        self.mode = mode or self.meta.get("mode", "fake")
        et = load_json(et_path)
        self.nodes = sorted(et["nodes"], key=lambda n: n["id"])
        self.byid = {n["id"]: n for n in self.nodes}
        self.kineto = load_json(kineto_path) if kineto_path else None
        self.warnings = []
        self._build_tree()
        self._map_kernels()

    # -- tree ---------------------------------------------------------------
    def _build_tree(self):
        self.parent = {}
        for n in self.nodes:
            p = n.get("ctrl_deps")
            if p is not None and p != n["id"] and p in self.byid:
                self.parent[n["id"]] = p
        self.is_op = {n["id"]: bool(_attr(n, "op_schema", "")) for n in self.nodes}
        self.is_root = {n["id"]: n["name"].startswith(ROOT_PREFIX) for n in self.nodes}
        # outermost op ancestor (or self) of every node; None if no op on path
        self.outer = {}
        for n in self.nodes:            # ids increase, parents come first
            i = n["id"]
            p = self.parent.get(i)
            po = self.outer.get(p) if p is not None else None
            if po is not None:
                self.outer[i] = po
            else:
                self.outer[i] = i if self.is_op[i] else None
        # annotation context: phase & enclosing record_function names
        self.ann = {}
        for n in self.nodes:
            i = n["id"]
            p = self.parent.get(i)
            ctx = self.ann.get(p, 0) if p is not None else 0
            nm = n["name"]
            if nm.startswith("autograd::engine::evaluate_function"):
                ctx |= 1
            elif nm.startswith("Optimizer.step") or nm.startswith("Optimizer.zero_grad"):
                ctx |= 2
            self.ann[i] = ctx

    def phase(self, i):
        c = self.ann.get(i, 0)
        return "opt" if c & 2 else ("bwd" if c & 1 else "fwd")

    # -- kernels --------------------------------------------------------------
    def _map_kernels(self):
        """kernel time per ET node id, via Kineto External id -> Record function id."""
        self.kern_us = collections.defaultdict(float)
        self.kern_n = collections.Counter()
        self.kern_streams = collections.defaultdict(collections.Counter)
        self.cpu_us = {}
        self.has_gpu = False
        self.unattributed_us = 0.0
        if not self.kineto:
            return
        rf_by_node = {}
        for n in self.nodes:
            rf = _attr(n, "rf_id")
            if rf is not None:
                rf_by_node[rf] = n["id"]
        ext2node = {}
        for e in self.kineto.get("traceEvents", []):
            cat = e.get("cat")
            if cat in ("cpu_op", "user_annotation"):
                a = e.get("args", {})
                rf = a.get("Record function id")
                nid = rf_by_node.get(rf)
                if nid is not None:
                    ext2node[a.get("External id")] = nid
                    self.cpu_us[nid] = e.get("dur", 0.0)
        for e in self.kineto.get("traceEvents", []):
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X":
                self.has_gpu = True
                a = e.get("args", {})
                nid = ext2node.get(a.get("External id"))
                if nid is None:
                    self.unattributed_us += e.get("dur", 0.0)
                    continue
                self.kern_us[nid] += e.get("dur", 0.0)
                self.kern_n[nid] += 1
                self.kern_streams[nid][a.get("stream")] += e.get("dur", 0.0)

    # -- classification -------------------------------------------------------
    def _comm_info(self, n):
        name = n["name"]
        ins, outs = n["inputs"], n["outputs"]
        ivals, itypes = ins.get("values", []), ins.get("types", [])
        ovals = outs.get("values", [])
        info = {"op": name}
        if name in C10D_OPS:
            coll, oi, ii = C10D_OPS[name]
            in_t = _tensors(ivals[ii]) if ii is not None and ii < len(ivals) else []
            out_t = _tensors(ivals[oi]) if oi is not None and oi < len(ivals) else []
            if not out_t and oi is not None:
                out_t = _tensors(ovals)
            dt = None
            for k in (ii, oi):
                if k is not None and k < len(itypes):
                    dt = dt or _dtype_of(itypes[k])
            info.update(coll=coll, functional=False, dtype=dt)
            if coll in ("SEND", "RECV") and len(ivals) > 2 and isinstance(ivals[2], int):
                info["peer_group_rank"] = ivals[2]
        else:
            short = name.split("::", 1)[1]
            coll = FUNC_OPS.get(short, "OTHER")
            in_t = _tensors(ivals[0]) if ivals else []
            out_t = _tensors(ovals)
            gname = None
            for v, t in zip(ivals, itypes):
                if isinstance(v, str) and t in ("String", "Any", "str") and not v in (
                        "sum", "avg", "max", "min", "product", "premul_sum"):
                    gname = v
            info.update(coll=coll, functional=True, dtype=_dtype_of(itypes[0] if itypes else ""),
                        group=gname)
        if coll in ("SEND",):
            out_t = []
        if coll in ("RECV",):
            in_t = []
        info["in_bytes"] = _nbytes(in_t)
        info["out_bytes"] = _nbytes(out_t) if coll not in ("ALL_REDUCE", "BROADCAST", "REDUCE") else _nbytes(in_t)
        if coll == "RECV":
            info["out_bytes"] = _nbytes(out_t)
            info["in_bytes"] = 0
        info["bytes"] = nccl_size(coll, info["in_bytes"], info["out_bytes"])
        info["out_storages"] = sorted({s for _, s, _, _ in (out_t or in_t)})
        return info

    # -- main pass ---------------------------------------------------------------
    def _subtree(self, root):
        if not hasattr(self, "_children"):
            ch = collections.defaultdict(list)
            for c, p in self.parent.items():
                ch[p].append(c)
            self._children = ch
        out, stack = [], [root]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(self._children.get(x, ()))
        return out

    def _is_comm(self, name):
        return (name in C10D_OPS or (name.startswith(FUNC_NS) and name not in WAIT_OPS)
                or (name.startswith("c10d::") and name != "c10d::record_param_comms"))

    def _calibrate(self):
        """Map stream keys recorded by commtag to Kineto stream ids."""
        self.calib = {}
        for n in self.nodes:
            if n["name"].startswith(CALIB_TAG):
                key = n["name"][len(CALIB_TAG):]
                st = collections.Counter()
                for x in self._subtree(n["id"]):
                    st.update(self.kern_streams.get(x, {}))
                if st:
                    self.calib[key] = st.most_common(1)[0][0]
        self.calib_identity = bool(self.calib) and all(
            str(v) == k for k, v in self.calib.items())
        if self.has_gpu and self.calib and not self.calib_identity:
            self.warnings.append(f"stream-id calibration mismatch {self.calib}; "
                                 "using calibrated keys only")
        if self.has_gpu and not self.calib:
            self.warnings.append("no stream calibration found; assuming driver stream ids "
                                 "== Kineto stream ids")

    def _map_stream(self, key):
        if key is None or key == "?":
            return None
        if not self.has_gpu:
            return "main"
        if key in self.calib:
            return self.calib[key]
        if (self.calib_identity or not self.calib) and key.isdigit():
            return int(key)
        return None

    def build(self, step_marker=None):
        """Build the fine-grained DAG for one profiled step."""
        use_gpu = self.has_gpu
        self._calibrate()
        in_step = None
        if step_marker:
            roots = [n["id"] for n in self.nodes if n["name"].startswith(step_marker)]
            if roots:
                in_step = set(self._subtree(roots[0]))
            else:
                self.warnings.append(f"step marker {step_marker!r} not found; using whole trace")

        def inside(i):
            return in_step is None or i in in_step

        # communication ops: outermost comm op on each path
        comm_ids = set()
        for n in self.nodes:
            i = n["id"]
            if not inside(i) or not self.is_op[i] or not self._is_comm(n["name"]):
                continue
            p, nested = self.parent.get(i), False
            while p is not None:
                if self.is_op.get(p) and self._is_comm(self.byid[p]["name"]):
                    nested = True
                    break
                p = self.parent.get(p)
            if not nested:
                comm_ids.add(i)

        def under_comm(nid):
            p = nid
            while p is not None:
                if p in comm_ids:
                    return True
                p = self.parent.get(p)
            return False

        def unit_of(nid):
            o = self.outer.get(nid)
            return o if o is not None else nid

        # compute units
        unit_ids = set()
        if use_gpu:
            for nid in self.kern_us:
                if inside(nid):
                    unit_ids.add(unit_of(nid))
        else:
            for n in self.nodes:
                i = n["id"]
                if inside(i) and self.is_op[i] and self.outer[i] == i:
                    unit_ids.add(i)
        unit_ids = {u for u in unit_ids if not under_comm(u) and u not in comm_ids}

        units, comms, items, tags = {}, [], [], collections.deque()
        readers = collections.defaultdict(list)
        stream_time = collections.Counter()
        for n in self.nodes:
            i = n["id"]
            if not inside(i):
                continue
            nm = n["name"]
            if nm.startswith(COMM_TAG):
                t = parse_tag(nm)
                if t:
                    t["pos"] = i
                    tags.append(t)
            elif nm.startswith(WAIT_TAG):
                d = parse_kv(nm, WAIT_TAG)
                items.append((i, "workwait", int(d["head"]), self._map_stream(d.get("stream"))))
            elif nm.startswith(EVT_TAG):
                d = parse_kv(nm, EVT_TAG)
                items.append((i, "ev" + d["head"], d.get("e"), self._map_stream(d.get("s"))))
            elif nm in WAIT_OPS:
                vals = n["inputs"].get("values", [])
                ts = _tensors(vals[0]) if vals else []
                items.append((i, "tensorwait", ts[0][1] if ts else None, None))

            if i in comm_ids:
                info = self._comm_info(n)
                info["pos"], info["phase"] = i, self.phase(i)
                if not info["functional"]:
                    want = info["coll"]
                    for k, t in enumerate(tags):
                        if want in _API_TO_COLL.get(t["api"], {want}):
                            info["tag"] = t
                            del tags[k]
                            break
                    t = info.get("tag")
                    info["group"] = t.get("pg") if t else None
                    info["async"] = t["async"] if t else False
                    if t:
                        info["seq"] = t["seq"]
                        if t.get("peer") is not None:
                            info["peer"] = t["peer"]
                        info["issue_stream"] = self._map_stream(t.get("stream"))
                else:
                    info["async"] = True
                sub = self._subtree(i)
                info["kernel_us"] = sum(self.kern_us.get(x, 0.0) for x in sub)
                info["cpu_us"] = self.cpu_us.get(i, 0.0)
                if info.get("issue_stream") is None and use_gpu:
                    st = collections.Counter()
                    for x in sub:
                        st.update(self.kern_streams.get(x, {}))
                    if st:     # fake-backend local copies run on the issuing stream
                        info["issue_stream"] = st.most_common(1)[0][0]
                comms.append(info)
                items.append((i, "comm", len(comms) - 1, info.get("issue_stream")))
            elif i in unit_ids:
                sub = self._subtree(i)
                if use_gpu:
                    us = sum(self.kern_us.get(x, 0.0) for x in sub)
                    st = collections.Counter()
                    for x in sub:
                        st.update(self.kern_streams.get(x, {}))
                    stream = st.most_common(1)[0][0] if st else None
                    nstreams = len(st)
                else:
                    us, stream, nstreams = self.cpu_us.get(i, 0.0), "main", 1
                units[i] = {"name": nm, "us": us, "phase": self.phase(i), "stream": stream,
                            "multi_stream": nstreams > 1,
                            "kernels": sum(self.kern_n.get(x, 0) for x in sub)}
                stream_time[stream] += us
                items.append((i, "comp", i, stream))
                for v in _tensors(n["inputs"].get("values", [])):
                    readers[v[1]].append(i)

        self.main_stream = stream_time.most_common(1)[0][0] if stream_time else "main"
        for u in units.values():
            if u["stream"] is None:
                u["stream"] = self.main_stream
        self.tags_unmatched = len(tags)
        self._units, self._comms, self._items, self._readers = units, comms, items, readers
        self._link()
        return self

    # -- dependency linking (stream frontier model) ------------------------------
    def _link(self):
        units, comms = self._units, self._comms
        main = self.main_stream
        F = collections.defaultdict(set)       # stream -> frontier node ids
        deps = collections.defaultdict(set)
        events = {}
        comm_q = {}
        pending_work = {}
        func_by_storage = collections.defaultdict(list)
        waited = set()
        self.node_stream = {}
        order = []

        for pos, kind, key, stream in sorted(self._items, key=lambda x: x[0]):
            s = stream if stream is not None else main
            if kind == "comp":
                n = f"c{key}"
                s = units[key]["stream"]
                deps[n] |= F[s]
                F[s] = {n}
                self.node_stream[n] = s
                order.append(n)
            elif kind == "comm":
                c = comms[key]
                n = f"m{key}"
                c["issue_stream"] = s
                deps[n] |= F[s]
                if c["coll"] in ("SEND", "RECV"):
                    # sends and receives to one peer progress independently (NCCL
                    # group semantics for batched isend/irecv); FIFO per direction
                    q = ("p2p", c.get("group"), c.get("peer"), c["coll"])
                else:
                    q = ("coll", c.get("group") if not c["functional"] else ("f", c.get("group")))
                c["comm_stream"] = "|".join(map(str, q))
                if q in comm_q:
                    deps[n].add(comm_q[q])
                comm_q[q] = n
                order.append(n)
                if not c.get("async", False):
                    F[s] = {n}
                    c["wait_kind"] = "sync"
                    waited.add(key)
                elif c["functional"]:
                    for st in c["out_storages"]:
                        func_by_storage[st].append(key)
                elif "seq" in c:
                    pending_work[c["seq"]] = key
            elif kind == "evrec":
                events[key] = set(F[s])
            elif kind == "evwait":
                F[s] |= events.get(key, set())
            elif kind == "workwait":
                k = pending_work.get(key)
                if k is not None:
                    F[s].add(f"m{k}")
                    comms[k]["wait_kind"] = "wait"
                    waited.add(k)
            elif kind == "tensorwait" and key is not None:
                cands = [k for k in func_by_storage.get(key, []) if comms[k]["pos"] < pos]
                if cands:
                    k = cands[-1]
                    F[main].add(f"m{k}")
                    comms[k].setdefault("wait_kind", "wait_tensor")
                    waited.add(k)

        # async ops never explicitly waited: first reader of their output
        for k, c in enumerate(comms):
            if k in waited:
                continue
            c["wait_kind"] = "none"
            if c["coll"] == "SEND":
                continue
            first = None
            for st in c["out_storages"]:
                rd = [u for u in self._readers.get(st, []) if u > c["pos"]]
                if rd:
                    first = min(rd) if first is None else min(first, min(rd))
            if first is not None:
                deps[f"c{first}"].add(f"m{k}")
                c["wait_kind"] = "data"

        def kind_of(a, b):
            ta, tb = a[0], b[0]
            if ta == "c" and tb == "c":
                return "stream" if self.node_stream[a] == self.node_stream[b] else "event"
            if ta == "c":
                return "issue"
            if tb == "c":
                return "complete"
            return "fifo" if comms[int(a[1:])]["comm_stream"] == comms[int(b[1:])]["comm_stream"] else "issue"

        edges = []
        for n in order:
            for d in sorted(deps[n], key=lambda x: (x[0], int(x[1:]))):
                if d != n:
                    edges.append((d, n, kind_of(d, n)))
        self.edges = edges
        self.order = order
        has_out = {d for d, _, _ in edges}
        self.no_consumer = [n for n in order if n not in has_out]

    # -- outputs ------------------------------------------------------------------
    def fine(self):
        nodes = []
        for n in self.order:
            if n[0] == "c":
                k = int(n[1:])
                u = self._units[k]
                nodes.append({"id": n, "type": "COMP", "name": u["name"], "phase": u["phase"],
                              "dur_us": round(u["us"], 3), "kernels": u["kernels"],
                              "stream": u["stream"], "pos": k})
            else:
                nodes.append(self._comm_node(int(n[1:])))
        return {"nodes": nodes, "edges": [list(e) for e in self.edges], "end_deps": self.no_consumer}

    def _comm_node(self, k):
        c = self._comms[k]
        topo = self.meta.get("topology", {}).get("groups", {})
        g = topo.get(str(c.get("group"))) if c.get("group") is not None else None
        peer = c.get("peer")
        if peer is None and "peer_group_rank" in c and g:
            r = c["peer_group_rank"]
            peer = g["ranks"][r] if 0 <= r < len(g["ranks"]) else None
        node = {"id": f"m{k}", "type": "COMM", "coll": c["coll"], "op": c["op"],
                "bytes": c["bytes"], "in_bytes": c["in_bytes"], "out_bytes": c["out_bytes"],
                "dtype": c.get("dtype"), "group": c.get("group"),
                "group_ranks": g["ranks"] if g else None,
                "axes": g["axes"] if g else None, "peer": peer,
                "async": c.get("async"), "wait": c.get("wait_kind"), "phase": c["phase"],
                "issue_stream": c.get("issue_stream"), "comm_stream": c.get("comm_stream"),
                "pos": c["pos"]}
        if self.mode == "real":
            node["measured_us"] = round(c["kernel_us"], 3)
        else:
            node["fake_kernel_us"] = round(c["kernel_us"], 3)
        return node

    def coarse(self):
        """Chain-compress compute units: merge a->b when both are compute on the
        same stream, b is a's only successor and a is b's only predecessor.
        This is exact (no dependency is added or removed)."""
        fine = self.fine()
        nodes = {n["id"]: n for n in fine["nodes"]}
        succ = collections.defaultdict(set)
        pred = collections.defaultdict(set)
        for a, b, _ in fine["edges"]:
            succ[a].add(b)
            pred[b].add(a)
        seg_of, segs = {}, []
        for n in fine["nodes"]:
            i = n["id"]
            if n["type"] != "COMP":
                continue
            ps = pred[i]
            if len(ps) == 1:
                (p,) = ps
                if (p in seg_of and nodes[p]["type"] == "COMP" and succ[p] == {i}
                        and nodes[p]["stream"] == n["stream"]):
                    seg_of[i] = seg_of[p]
                    segs[seg_of[i]].append(i)
                    continue
            seg_of[i] = len(segs)
            segs.append([i])
        seg_nodes = []
        for si, members in enumerate(segs):
            ms = [nodes[m] for m in members]
            ph, top = collections.Counter(), collections.Counter()
            for m in ms:
                ph[m["phase"]] += m["dur_us"] + 1e-9
                top[m["name"]] += m["dur_us"]
            seg_nodes.append({"id": f"s{si}", "type": "COMP", "phase": ph.most_common(1)[0][0],
                              "dur_us": round(sum(m["dur_us"] for m in ms), 3),
                              "n_ops": len(ms), "kernels": sum(m["kernels"] for m in ms),
                              "stream": ms[0]["stream"],
                              "top_ops": [[k, round(v, 3)] for k, v in top.most_common(4)]})

        def remap(x):
            return f"s{seg_of[x]}" if x in seg_of else x

        es, seen = [], set()
        for a, b, k in fine["edges"]:
            a2, b2 = remap(a), remap(b)
            if a2 == b2 or (a2, b2) in seen:
                continue
            seen.add((a2, b2))
            es.append([a2, b2, k])
        out, emitted = [], set()
        for n in fine["nodes"]:
            if n["type"] == "COMP":
                s = seg_of[n["id"]]
                if s not in emitted:
                    emitted.add(s)
                    out.append(seg_nodes[s])
            else:
                out.append(n)
        has_out = {a for a, _, _ in es}
        return {"nodes": out, "edges": es, "end_deps": [n["id"] for n in out if n["id"] not in has_out]}

    def signature(self, dag=None):
        """Structure fingerprint: comm sequence + compute segment count."""
        dag = dag or self.coarse()
        comm = [(n["coll"], ",".join(n["axes"] or []) if n.get("axes") else str(n.get("group")),
                 n["bytes"], n.get("peer") is not None) for n in dag["nodes"] if n["type"] == "COMM"]
        n_comp = sum(1 for n in dag["nodes"] if n["type"] == "COMP")
        h = hashlib.sha256(json.dumps([comm, n_comp, sorted(map(tuple, dag["edges"]))]).encode()).hexdigest()
        return {"sha256": h, "n_comm": len(comm), "n_comp_segments": n_comp,
                "n_edges": len(dag["edges"])}

    def summary(self, dag):
        comp = [n for n in dag["nodes"] if n["type"] == "COMP"]
        comm = [n for n in dag["nodes"] if n["type"] == "COMM"]
        by = collections.defaultdict(lambda: [0, 0])
        for c in comm:
            k = (c["coll"], ",".join(c["axes"] or ["?"]))
            by[k][0] += 1
            by[k][1] += c["bytes"]
        return {
            "compute_us": round(sum(n["dur_us"] for n in comp), 3),
            "compute_us_by_phase": {p: round(sum(n["dur_us"] for n in comp if n["phase"] == p), 3)
                                    for p in ("fwd", "bwd", "opt")},
            "n_comp_segments": len(comp), "n_comm": len(comm),
            "comm_by_type_axis": {f"{k[0]}@{k[1]}": {"count": v[0], "bytes": v[1]}
                                  for k, v in sorted(by.items())},
            "untagged_c10d_comms": sum(1 for c in comm if c["group"] is None),
            "streams": sorted({str(n.get("stream")) for n in comp}),
            "unmatched_tags": getattr(self, "tags_unmatched", 0),
            "unattributed_gpu_us": round(self.unattributed_us, 3),
            "time_source": "gpu_kernels" if self.has_gpu else "cpu_ops",
        }


# ----------------------------------------------------------------- driver

def extract_rank_dir(rank_dir, out_dir=None, write_fine=True):
    """Process every step<k>.et.json in a rank<R> directory."""
    meta_path = os.path.join(rank_dir, "meta.json")
    meta = load_json(meta_path) if os.path.exists(meta_path) else {}
    out_dir = out_dir or rank_dir
    os.makedirs(out_dir, exist_ok=True)
    steps = sorted(int(m.group(1)) for f in os.listdir(rank_dir)
                   if (m := re.match(r"step(\d+)\.et\.json$", f)))
    results = []
    for k in steps:
        et = os.path.join(rank_dir, f"step{k}.et.json")
        kin = os.path.join(rank_dir, f"step{k}.kineto.json")
        rt = RankTrace(et, kin if os.path.exists(kin) else None, meta)
        rt.build(step_marker=f"dagtrace::step#{k}")
        dag = rt.coarse()
        doc = {"format": "dagtrace.rank_dag/1", "rank": meta.get("rank"), "step": k,
               "mode": rt.mode, "world_size": meta.get("world_size"),
               "parallel_dims": meta.get("parallel_dims"),
               "summary": rt.summary(dag), "signature": rt.signature(dag),
               "warnings": rt.warnings, **dag}
        p = os.path.join(out_dir, f"step{k}.dag.json")
        with open(p, "w") as f:
            json.dump(doc, f)
        if write_fine:
            with gzip.open(os.path.join(out_dir, f"step{k}.fine.json.gz"), "wt") as f:
                json.dump({"rank": meta.get("rank"), "step": k, **rt.fine()}, f)
        results.append((k, p, doc))
    return results


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Extract compute/comm DAGs from captured traces")
    ap.add_argument("trace_dir", help="output dir of titan_capture (contains rank<R>/)")
    ap.add_argument("--no-fine", action="store_true")
    a = ap.parse_args(argv)
    rank_dirs = sorted(d for d in os.listdir(a.trace_dir) if re.match(r"rank\d+$", d))
    for rd in rank_dirs:
        for k, p, doc in extract_rank_dir(os.path.join(a.trace_dir, rd), write_fine=not a.no_fine):
            s = doc["summary"]
            print(f"{rd} step{k}: {s['n_comp_segments']} compute segments "
                  f"({s['compute_us']/1e3:.2f} ms, {s['time_source']}), {s['n_comm']} comms, "
                  f"sig {doc['signature']['sha256'][:12]} -> {p}")
            for t, v in s["comm_by_type_axis"].items():
                print(f"    {t:40s} x{v['count']:5d}  {v['bytes']/2**20:10.2f} MiB")
            if s["untagged_c10d_comms"]:
                print(f"    WARNING: {s['untagged_c10d_comms']} comm ops without a process group")


if __name__ == "__main__":
    main()
