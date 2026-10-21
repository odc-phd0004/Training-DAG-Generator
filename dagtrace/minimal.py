"""Export minimal DAGs: vertices = compute / collective / p2p, edges = dependencies.

  python -m dagtrace.minimal traces/<name>

writes
  traces/<name>/minimal/gpu<r>.json   one per GPU of the job
  traces/<name>/minimal/job.json      one DAG for the whole job

Per-GPU DAG vertices:
  {"id": 0, "type": "compute",    "time_us": 812.4}
  {"id": 1, "type": "collective", "op": "all_gather", "bytes": 2097152, "group": [0, 1]}
  {"id": 2, "type": "p2p",        "op": "send", "bytes": 1048576, "peer": 4}

Job DAG vertices:
  compute    -> {"id", "type": "compute", "gpu", "time_us"}
  collective -> {"id", "type": "collective", "op", "bytes", "group"}  ONE vertex per
                collective instance, shared by all its member GPUs
  p2p        -> {"id", "type": "p2p", "bytes", "src", "dst"}  ONE vertex per send/recv pair
Edges: [u, v] means v cannot start before u finishes.
"""
from __future__ import annotations

import json
import os

from dagtrace.extract import load_json


def _vertex(n):
    if n["type"] == "COMP":
        return {"type": "compute", "time_us": n["dur_us"]}
    if n["coll"] in ("SEND", "RECV"):
        return {"type": "p2p", "op": n["coll"].lower(), "bytes": n["bytes"], "peer": n.get("peer")}
    return {"type": "collective", "op": n["coll"].lower(), "bytes": n["bytes"],
            "group": n.get("group_ranks")}


def export(trace_dir, out_dir=None):
    g = load_json(os.path.join(trace_dir, "global.dag.json"))
    out_dir = out_dir or os.path.join(trace_dir, "minimal")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- per-GPU DAGs
    by_rank = {}
    for n in g["nodes"]:
        by_rank.setdefault(n["rank"], []).append(n)
    edges_by_rank = {}
    for a, b, _ in g["edges"]:
        edges_by_rank.setdefault(int(a.split(":")[0]), []).append((a, b))
    for r, nodes in by_rank.items():
        idx = {n["id"]: i for i, n in enumerate(nodes)}
        doc = {"gpu": r, "vertices": [{"id": i, **_vertex(n)} for i, n in enumerate(nodes)],
               "edges": sorted({(idx[a], idx[b]) for a, b in edges_by_rank.get(r, [])
                                if a in idx and b in idx})}
        with open(os.path.join(out_dir, f"gpu{r}.json"), "w") as f:
            json.dump(doc, f)

    # ---------- job DAG: merge each collective / p2p instance into one vertex
    vid, vertices = {}, []
    inst_vid = {}
    for c in g["collectives"]:
        inst_vid[c["id"]] = len(vertices)
        vertices.append({"id": len(vertices), "type": "collective", "op": c["coll"].lower(),
                         "bytes": c["bytes"], "group": c["ranks"]})
    for p in g["p2p"]:
        inst_vid[p["id"]] = len(vertices)
        vertices.append({"id": len(vertices), "type": "p2p", "bytes": p["bytes"],
                         "src": p["src"], "dst": p["dst"]})
    for n in g["nodes"]:
        if n["type"] == "COMM" and n.get("instance") in inst_vid:
            vid[n["id"]] = inst_vid[n["instance"]]
        else:
            vid[n["id"]] = len(vertices)
            v = {"id": len(vertices), **_vertex(n)}
            if v["type"] == "compute":
                v["gpu"] = n["rank"]
            else:   # unmatched / single-member comm
                v["gpu"] = n["rank"]
            vertices.append(v)
    edges = sorted({(vid[a], vid[b]) for a, b, _ in g["edges"]
                    if a in vid and b in vid and vid[a] != vid[b]})
    job = {"world_size": g["world_size"], "parallel_dims": g.get("parallel_dims"),
           "vertices": vertices, "edges": edges}
    with open(os.path.join(out_dir, "job.json"), "w") as f:
        json.dump(job, f)
    return out_dir, len(by_rank), len(vertices), len(edges)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Export minimal per-GPU and job DAGs")
    ap.add_argument("trace_dir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out, ngpu, nv, ne = export(a.trace_dir, a.out)
    print(f"wrote {ngpu} per-GPU DAGs and job.json ({nv} vertices, {ne} edges) -> {out}")


if __name__ == "__main__":
    main()
