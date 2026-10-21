"""Stitch per-rank DAGs into the DAG of the whole N-GPU job.

Only one rank per pipeline stage needs to be captured: in SPMD training every
rank of the same PP stage runs the same program on a different shard.  For a
rank x that was not captured, we take the captured rank r of the same PP
stage as a template and relabel every process group: "r's TP group" becomes
"the TP group containing x", using the complete group partitions recorded in
meta.json.  P2P peers are relabelled by their index inside the group.

Then collectives are matched across ranks.  NCCL requires the members of a
communicator to issue its collectives in the same order, so the i-th
collective on group G is the same collective on every member of G.  For every
match we check that the members agree on the collective type and size; for
P2P we check that the k-th SEND a->b has the same size as the k-th RECV b<-a.
Any disagreement means the per-rank programs are inconsistent and is reported
-- this is a built-in cross-check of the capture.

Output: global.step<k>.dag.json with
  nodes        per-rank COMP and COMM nodes ("<rank>:<local id>")
  collectives  one entry per collective instance with its member nodes
  p2p          one entry per send/recv pair
  edges        per-rank dependency edges (rank-prefixed)
"""
from __future__ import annotations

import collections
import json
import os
import re

from dagtrace.extract import load_json


def _rank_dirs(trace_dir):
    out = {}
    for d in os.listdir(trace_dir):
        m = re.match(r"rank(\d+)$", d)
        if m:
            out[int(m.group(1))] = os.path.join(trace_dir, d)
    return dict(sorted(out.items()))


class Topology:
    def __init__(self, meta):
        self.meta = meta
        self.world = meta["world_size"]
        self.rank = meta["rank"]
        self.groups = meta["topology"]["groups"]

    def axis_partition(self, axis):
        for g in self.groups.values():
            if axis in g["axes"] and g.get("partition"):
                return g["partition"]
        return None

    def pp_coord(self, x):
        part = self.axis_partition("pp")
        if not part:
            return 0
        for blk in part:
            if x in blk:
                return blk.index(x)
        raise KeyError(x)

    def block_for(self, gname, x):
        g = self.groups.get(str(gname))
        if g is None:
            return None
        if g.get("partition"):
            for blk in g["partition"]:
                if x in blk:
                    return blk
            return None
        if len(g["ranks"]) == 1:
            return [x]
        return None

    def translate_peer(self, gname, peer, x):
        """peer is a global rank in the template rank's group gname."""
        g = self.groups.get(str(gname)) if gname is not None else None
        cands = [g] if g and g.get("partition") else []
        cands += [h for h in self.groups.values() if h.get("partition")]
        for h in cands:
            for blk in h["partition"]:
                if self.rank in blk and peer in blk:
                    j = blk.index(peer)
                    for blk2 in h["partition"]:
                        if x in blk2:
                            return blk2[j]
        return None


def stitch(trace_dir, step=None, world_ranks=None, out_path=None, verbose=True):
    rdirs = _rank_dirs(trace_dir)
    if not rdirs:
        raise SystemExit(f"no rank<R> directories in {trace_dir}")
    templates = {}
    for r, d in rdirs.items():
        meta = load_json(os.path.join(d, "meta.json"))
        steps = sorted(int(m.group(1)) for f in os.listdir(d)
                       if (m := re.match(r"step(\d+)\.dag\.json$", f)))
        if not steps:
            raise SystemExit(f"{d}: run `python -m dagtrace.extract {trace_dir}` first")
        k = step if step is not None else steps[-1]
        dag = load_json(os.path.join(d, f"step{k}.dag.json"))
        templates[r] = (Topology(meta), dag, k)
    any_topo = next(iter(templates.values()))[0]
    world = any_topo.world
    by_stage = {}
    for r, (topo, dag, k) in templates.items():
        by_stage.setdefault(topo.pp_coord(r), r)
    pp_part = any_topo.axis_partition("pp")
    pp = len(pp_part[0]) if pp_part else 1
    missing = [s for s in range(pp) if s not in by_stage]
    if missing:
        raise SystemExit(f"missing captures for PP stages {missing}; run titan_capture "
                         f"with --pp-rank for each of them")
    ranks = world_ranks if world_ranks is not None else list(range(world))

    nodes, edges = [], []
    per_rank_coll = collections.defaultdict(lambda: collections.defaultdict(list))
    sends = collections.defaultdict(list)       # (src,dst) -> [node]
    recvs = collections.defaultdict(list)
    problems = []
    for x in ranks:
        topo_any = any_topo
        stage = topo_any.pp_coord(x)
        r = by_stage[stage]
        topo, dag, k = templates[r]
        pre = f"{x}:"
        for n in dag["nodes"]:
            m = dict(n)
            m["id"] = pre + n["id"]
            m["rank"] = x
            if n["type"] == "COMM":
                gr = topo.block_for(n.get("group"), x) if n.get("group") is not None else None
                if gr is None and n.get("group_ranks") and len(n["group_ranks"]) == 1:
                    gr = [x]
                if gr is None:
                    problems.append(f"rank {x}: cannot map group {n.get('group')} of {n['id']}")
                    gr = n.get("group_ranks") or [x]
                m["group_ranks"] = gr
                if n.get("peer") is not None:
                    m["peer"] = topo.translate_peer(n.get("group"), n["peer"], x)
                    if m["peer"] is None:
                        problems.append(f"rank {x}: cannot map peer of {n['id']}")
                for key in ("pos", "issue_stream"):
                    m.pop(key, None)
                if n["coll"] == "SEND":
                    sends[(x, m["peer"])].append(m)
                elif n["coll"] == "RECV":
                    recvs[(m["peer"], x)].append(m)
                else:
                    per_rank_coll[frozenset(gr)][x].append(m)
            else:
                m.pop("top_ops", None)
            nodes.append(m)
        for a, b, kind in dag["edges"]:
            edges.append([pre + a, pre + b, kind])

    collectives = []
    for G, members in per_rank_coll.items():
        lens = {x: len(v) for x, v in members.items()}
        present = set(members)
        if len(G) > 1 and present != (set(G) & set(ranks)):
            problems.append(f"group {sorted(G)}: only ranks {sorted(present)} issued collectives")
        if len(set(lens.values())) > 1:
            problems.append(f"group {sorted(G)}: ranks issue different numbers of collectives {lens}")
        for i in range(max(lens.values())):
            mem = {x: v[i] for x, v in members.items() if i < len(v)}
            sig = {(m["coll"], m["bytes"]) for m in mem.values()}
            if len(sig) > 1:
                problems.append(f"group {sorted(G)} collective #{i}: members disagree {sig}")
            m0 = next(iter(mem.values()))
            cid = f"C{len(collectives)}"
            for m in mem.values():
                m["instance"] = cid
            collectives.append({"id": cid, "coll": m0["coll"], "bytes": m0["bytes"],
                                "dtype": m0.get("dtype"), "axes": m0.get("axes"),
                                "ranks": sorted(G), "phase": m0.get("phase"),
                                "members": {str(x): m["id"] for x, m in mem.items()}})
    p2p = []
    for key in sorted(set(sends) | set(recvs), key=str):
        s, rv = sends.get(key, []), recvs.get(key, [])
        if len(s) != len(rv):
            problems.append(f"p2p {key[0]}->{key[1]}: {len(s)} sends vs {len(rv)} recvs")
        for i in range(min(len(s), len(rv))):
            if s[i]["bytes"] != rv[i]["bytes"]:
                problems.append(f"p2p {key} #{i}: send {s[i]['bytes']}B != recv {rv[i]['bytes']}B")
            pid = f"P{len(p2p)}"
            s[i]["instance"] = rv[i]["instance"] = pid
            p2p.append({"id": pid, "src": key[0], "dst": key[1], "bytes": s[i]["bytes"],
                        "send": s[i]["id"], "recv": rv[i]["id"], "phase": s[i].get("phase")})

    doc = {"format": "dagtrace.global_dag/1", "world_size": world, "ranks": ranks,
           "step": {str(r): v[2] for r, v in templates.items()},
           "templates": {str(s): r for s, r in by_stage.items()},
           "parallel_dims": any_topo.meta.get("parallel_dims"),
           "model": any_topo.meta.get("model", {}).get("name"),
           "consistency": {"ok": not problems, "problems": problems[:200],
                           "n_problems": len(problems)},
           "counts": {"nodes": len(nodes), "edges": len(edges),
                      "collectives": len(collectives), "p2p": len(p2p)},
           "nodes": nodes, "edges": edges, "collectives": collectives, "p2p": p2p}
    out_path = out_path or os.path.join(trace_dir, "global.dag.json")
    with open(out_path, "w") as f:
        json.dump(doc, f)
    if verbose:
        print(f"global DAG: {world} ranks, {len(nodes)} nodes, {len(edges)} edges, "
              f"{len(collectives)} collectives, {len(p2p)} p2p transfers -> {out_path}")
        print("consistency:", "OK" if not problems else f"{len(problems)} PROBLEMS")
        for p in problems[:20]:
            print("  ", p)
    return doc


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace_dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    stitch(a.trace_dir, a.step, out_path=a.out)


if __name__ == "__main__":
    main()
