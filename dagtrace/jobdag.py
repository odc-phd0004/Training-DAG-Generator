"""Assemble per-GPU DAGs (from direct capture) into one job DAG.

Per-GPU DAG  rank<r>/gpu_dag.step<k>.json
  nodes: {"id", "type": compute|collective|p2p, "dur_us" | "op","bytes","group","peer"}
  edges: [[u, v], ...]          u must finish before v starts

Job DAG      job_dag.json
  vertices:
    compute     "c:<rank>:<id>"   one per compute segment per GPU   {rank, dur_us}
    collective  "C<k>"            ONE vertex shared by all members  {op, bytes, ranks}
    p2p         "P<k>"            ONE vertex per send/recv pair     {bytes, src, dst}
  edges: dependencies only

Scale: only one rank per pipeline stage is captured; every other rank of that
stage runs the same program on another shard.  job_dag.compact.json stores just
those templates plus the process-group partitions (size independent of the
number of GPUs).  The explicit job DAG is expanded from it (by default only up
to --max-expand GPUs), and `gpu_dag(compact, x)` gives any rank's DAG on demand.

Matching: NCCL requires every member of a communicator to issue its
collectives in the same order, so the i-th collective on group G is the same
collective on all members; the k-th send a->b pairs with the k-th recv b<-a.
Members must agree on op and bytes -- any disagreement is reported.

    python -m dagtrace.jobdag traces/<name> [--max-expand 1024] [--rank 37]
"""
from __future__ import annotations

import collections
import json
import os
import re

from dagtrace.extract import load_json


# ---------------------------------------------------------------- compact form
def build_compact(trace_dir, step=None):
    templates, groups_axes = {}, {}
    world, pp_part = None, None
    for d in sorted(os.listdir(trace_dir)):
        m = re.match(r"rank(\d+)$", d)
        if not m:
            continue
        rd = os.path.join(trace_dir, d)
        steps = sorted(int(x.group(1)) for f in os.listdir(rd)
                       if (x := re.match(r"gpu_dag\.step(\d+)\.json$", f)))
        if not steps:
            continue
        k = step if step is not None else steps[-1]
        g = load_json(os.path.join(rd, f"gpu_dag.step{k}.json"))
        meta = load_json(os.path.join(rd, "meta.json"))
        topo = meta["topology"]["groups"]
        r = g["rank"]
        world = g["world_size"]
        # replace process-group names by (axis, partition) so templates are portable
        part_of = {}
        for gname, info in topo.items():
            if info.get("partition"):
                key = ",".join(info["axes"]) or f"pg{gname}"
                groups_axes[key] = info["partition"]
                part_of[gname] = key
                if "pp" in info["axes"]:
                    pp_part = info["partition"]
        nodes = []
        for n in g["nodes"]:
            n = dict(n)
            if n["type"] != "compute":
                gname = n.pop("group", None)
                if gname in part_of:
                    n["axis"] = part_of[gname]
                else:      # group without a mesh partition (e.g. size-1)
                    n["axis"] = None
                    n["fixed_ranks"] = g.get("groups", {}).get(gname)
            nodes.append(n)
        stage = _coord(pp_part, r) if pp_part else 0
        templates[str(stage)] = {"rank": r, "step": g["step"], "nodes": nodes, "edges": g["edges"],
                                 "signature": g.get("signature")}
    if not templates:
        raise SystemExit(f"no rank*/gpu_dag.step*.json in {trace_dir}")
    npp = len(pp_part[0]) if pp_part else 1
    missing = [s for s in range(npp) if str(s) not in templates]
    return {"format": "dagtrace.job_compact/1", "world_size": world, "pp": npp,
            "missing_stages": missing, "axes": groups_axes, "templates": templates}


def _coord(partition, x):
    for blk in partition:
        if x in blk:
            return blk.index(x)
    raise KeyError(x)


def _block(partition, x):
    for blk in partition:
        if x in blk:
            return blk
    return None


def _template_for(compact, x):
    pp = compact["axes"].get("pp") or next((v for k, v in compact["axes"].items()
                                            if "pp" in k.split(",")), None)
    s = _coord(pp, x) if pp else 0
    return compact["templates"][str(s)]


def gpu_dag(compact, x):
    """The per-GPU DAG of any rank x, with concrete group ranks / peers."""
    t = _template_for(compact, x)
    r = t["rank"]
    nodes = []
    for n in t["nodes"]:
        m = dict(n)
        if n["type"] != "compute":
            ax = n.get("axis")
            if ax:
                m["ranks"] = _block(compact["axes"][ax], x)
                if n.get("peer") is not None:
                    src_blk = _block(compact["axes"][ax], r)
                    m["peer"] = m["ranks"][src_blk.index(n["peer"])]
            else:
                m["ranks"] = [x] if (n.get("fixed_ranks") or [r]) == [r] else n.get("fixed_ranks")
        nodes.append(m)
    return {"rank": x, "world_size": compact["world_size"], "nodes": nodes, "edges": t["edges"]}


# --------------------------------------------------------------- explicit form
def expand(compact, ranks=None):
    world = compact["world_size"]
    ranks = list(range(world)) if ranks is None else ranks
    verts, edges, problems = [], [], []
    coll = collections.defaultdict(lambda: collections.defaultdict(list))  # G -> rank -> [vid]
    sends = collections.defaultdict(list)
    recvs = collections.defaultdict(list)
    local = {}       # (rank, local id) -> vertex id or placeholder
    for x in ranks:
        g = gpu_dag(compact, x)
        for n in g["nodes"]:
            if n["type"] == "compute":
                vid = f"c:{x}:{n['id']}"
                verts.append({"id": vid, "type": "compute", "rank": x, "dur_us": n["dur_us"]})
                local[(x, n["id"])] = vid
            elif n["type"] == "p2p":
                ref = {"rank": x, "lid": n["id"], "op": n["op"], "bytes": n["bytes"], "peer": n["peer"]}
                (sends if n["op"] == "send" else recvs)[(x, n["peer"]) if n["op"] == "send"
                                                       else (n["peer"], x)].append(ref)
            else:
                G = tuple(n["ranks"] or [x])
                coll[G][x].append({"rank": x, "lid": n["id"], "op": n["op"], "bytes": n["bytes"]})
        for u, v in g["edges"]:
            edges.append(((x, u), (x, v)))
    for G, members in coll.items():
        present = set(members)
        want = set(G) & set(ranks)
        if present != want:
            problems.append(f"group {list(G)[:8]}{'...' if len(G) > 8 else ''}: "
                            f"{len(want - present)} members issued nothing")
        lens = {len(v) for v in members.values()}
        if len(lens) > 1:
            problems.append(f"group {list(G)[:8]}: members issue different numbers of collectives")
        for i in range(max(lens)):
            mem = [v[i] for v in members.values() if i < len(v)]
            sig = {(m["op"], m["bytes"]) for m in mem}
            if len(sig) > 1:
                problems.append(f"group {list(G)[:8]} collective #{i}: members disagree {sig}")
            vid = f"C{len(verts)}"
            verts.append({"id": vid, "type": "collective", "op": mem[0]["op"],
                          "bytes": mem[0]["bytes"], "ranks": list(G)})
            for m in mem:
                local[(m["rank"], m["lid"])] = vid
    for key in sorted(set(sends) | set(recvs), key=str):
        s, rv = sends.get(key, []), recvs.get(key, [])
        if len(s) != len(rv):
            problems.append(f"p2p {key[0]}->{key[1]}: {len(s)} sends vs {len(rv)} recvs")
        for i in range(min(len(s), len(rv))):
            if s[i]["bytes"] != rv[i]["bytes"]:
                problems.append(f"p2p {key} #{i}: send {s[i]['bytes']} B != recv {rv[i]['bytes']} B")
            vid = f"P{len(verts)}"
            verts.append({"id": vid, "type": "p2p", "bytes": s[i]["bytes"],
                          "src": key[0], "dst": key[1]})
            local[(s[i]["rank"], s[i]["lid"])] = vid
            local[(rv[i]["rank"], rv[i]["lid"])] = vid
    out_edges, seen = [], set()
    for a, b in edges:
        u, v = local.get(a), local.get(b)
        if u is None or v is None or u == v or (u, v) in seen:
            continue
        seen.add((u, v))
        out_edges.append([u, v])
    return {"format": "dagtrace.job/1", "world_size": world, "ranks": len(ranks),
            "consistency": {"ok": not problems, "n_problems": len(problems), "problems": problems[:100]},
            "counts": collections.Counter(v["type"] for v in verts) | {"edges": len(out_edges)},
            "vertices": verts, "edges": out_edges}


def critical_path(job, comm_us=None):
    """Longest path with compute times and optional comm-time function."""
    comm_us = comm_us or (lambda v: 0.0)
    verts = {v["id"]: v for v in job["vertices"]}
    succ = collections.defaultdict(list)
    indeg = collections.Counter()
    for a, b in job["edges"]:
        succ[a].append(b)
        indeg[b] += 1
    start = collections.defaultdict(float)
    q = collections.deque(v for v in verts if indeg[v] == 0)
    fin, done = {}, 0
    while q:
        v = q.popleft()
        d = verts[v]["dur_us"] if verts[v]["type"] == "compute" else comm_us(verts[v])
        fin[v] = start[v] + d
        done += 1
        for w in succ[v]:
            start[w] = max(start[w], fin[v])
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return (max(fin.values()) if fin else 0.0), done == len(verts)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build the job DAG from per-GPU DAGs")
    ap.add_argument("trace_dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--max-expand", type=int, default=1024,
                    help="write the explicit job DAG only up to this many GPUs")
    ap.add_argument("--rank", type=int, action="append",
                    help="also write the per-GPU DAG of this rank (repeatable)")
    ap.add_argument("--all-ranks", action="store_true",
                    help="write per_gpu/gpu_dag.rank<x>.json for every rank")
    a = ap.parse_args(argv)
    c = build_compact(a.trace_dir, a.step)
    if c["missing_stages"]:
        raise SystemExit(f"capture missing for pipeline stages {c['missing_stages']}")
    p = os.path.join(a.trace_dir, "job_dag.compact.json")
    json.dump(c, open(p, "w"))
    print(f"compact job DAG ({c['world_size']} GPUs, {c['pp']} stage template(s)) -> {p}")
    for x in a.rank or []:
        q = os.path.join(a.trace_dir, f"gpu_dag.rank{x}.json")
        json.dump(gpu_dag(c, x), open(q, "w"))
        print(f"per-GPU DAG of rank {x} -> {q}")
    if a.all_ranks:
        d = os.path.join(a.trace_dir, "per_gpu")
        os.makedirs(d, exist_ok=True)
        for x in range(c["world_size"]):
            json.dump(gpu_dag(c, x), open(os.path.join(d, f"gpu_dag.rank{x}.json"), "w"))
        print(f"per-GPU DAGs of all {c['world_size']} ranks -> {d}/")
    if c["world_size"] <= a.max_expand:
        job = expand(c)
        p = os.path.join(a.trace_dir, "job_dag.json")
        json.dump(job, open(p, "w"))
        cp, ok = critical_path(job)
        print(f"job DAG: {dict(job['counts'])} -> {p}")
        print(f"consistency: {'OK' if job['consistency']['ok'] else job['consistency']['problems'][:5]}")
        print(f"critical path with zero-cost communication: {cp/1e3:.3f} ms"
              + ("" if ok else "  (WARNING: cycle / unmatched vertex)"))
    else:
        print(f"{c['world_size']} GPUs > --max-expand {a.max_expand}: explicit job DAG not written "
              "(use the compact form, or dagtrace.jobdag.expand(compact, ranks=[...]))")


if __name__ == "__main__":
    main()
