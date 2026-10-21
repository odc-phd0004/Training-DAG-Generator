"""Annotate a global DAG with estimated communication times and replay it.

The communication model is deliberately simple and transparent (it is the
part a networking study replaces with its own model or a packet-level
simulator such as ASTRA-sim/ns-3):

  ring-style alpha-beta model, nccl-tests "bus bandwidth" conventions
    ALL_REDUCE       t = 2(n-1)/n * S / B + 2(n-1) * a
    ALL_GATHER / REDUCE_SCATTER / ALL_TO_ALL
                     t =  (n-1)/n * S / B +  (n-1) * a
    BROADCAST/REDUCE t = S / B + log2(n) * a
    SEND/RECV        t = S / B + a
  B and a are the intra-node values if every member is on one node, else the
  inter-node values (ranks are packed node by node, gpus_per_node at a time).

Replay semantics (no network contention -- each transfer gets its full
bandwidth; contention is exactly what a multi-tenant study adds):
  * a node starts when all its dependencies have finished
  * a collective instance starts when every member's COMM node is ready
  * a P2P transfer starts when both the SEND and the RECV node are ready
  * COMP nodes take their measured GPU time
"""
from __future__ import annotations

import collections
import heapq
import json
import math

from dagtrace.extract import load_json

DEFAULT_CLUSTER = {
    "gpus_per_node": 8,
    "intra_bw_GBps": 300.0,     # NVLink-class per-GPU bus bandwidth
    "inter_bw_GBps": 45.0,      # ~400 Gb/s NIC per GPU, ~90% efficiency
    "intra_alpha_us": 3.0,
    "inter_alpha_us": 8.0,
}


def comm_time_us(coll, nbytes, ranks, cluster):
    n = len(ranks)
    if n <= 1 or nbytes == 0 and coll not in ("BARRIER",):
        return 0.0
    gpn = cluster["gpus_per_node"]
    multi = len({r // gpn for r in ranks}) > 1
    bw = (cluster["inter_bw_GBps"] if multi else cluster["intra_bw_GBps"]) * 1e3  # bytes/us
    a = cluster["inter_alpha_us"] if multi else cluster["intra_alpha_us"]
    if coll == "ALL_REDUCE":
        return 2 * (n - 1) / n * nbytes / bw + 2 * (n - 1) * a
    if coll in ("ALL_GATHER", "REDUCE_SCATTER", "ALL_TO_ALL", "GATHER", "SCATTER"):
        return (n - 1) / n * nbytes / bw + (n - 1) * a
    if coll in ("BROADCAST", "REDUCE"):
        return nbytes / bw + math.ceil(math.log2(n)) * a
    if coll in ("SEND", "RECV"):
        return nbytes / bw + a
    if coll == "BARRIER":
        return 2 * (n - 1) * a
    return (n - 1) / n * nbytes / bw + (n - 1) * a


def annotate(doc, cluster):
    for c in doc["collectives"]:
        c["est_us"] = round(comm_time_us(c["coll"], c["bytes"], c["ranks"], cluster), 3)
    for p in doc["p2p"]:
        p["est_us"] = round(comm_time_us("SEND", p["bytes"], [p["src"], p["dst"]], cluster), 3)
    doc["cluster"] = cluster
    return doc


def replay(doc):
    nodes = {n["id"]: n for n in doc["nodes"]}
    preds = collections.defaultdict(set)
    succs = collections.defaultdict(list)
    for a, b, _ in doc["edges"]:
        if a in nodes and b in nodes:
            preds[b].add(a)
            succs[a].append(b)
    inst = {}
    for c in doc["collectives"]:
        inst[c["id"]] = ("coll", c, list(c["members"].values()))
    for p in doc["p2p"]:
        inst[p["id"]] = ("p2p", p, [p["send"], p["recv"]])
    remaining = {i: len(preds[i]) for i in nodes}
    ready_t = collections.defaultdict(float)
    finish = {}
    inst_ready = collections.defaultdict(dict)
    heap = []

    def push(nid, t):
        heapq.heappush(heap, (t, nid))

    def node_ready(nid, t):
        n = nodes[nid]
        if n["type"] == "COMP":
            push(nid, t + n["dur_us"])
            return
        iid = n.get("instance")
        if iid is None or iid not in inst:    # unmatched / single-rank comm
            push(nid, t)
            return
        kind, rec, members = inst[iid]
        inst_ready[iid][nid] = t
        if len(inst_ready[iid]) == len(members):
            start = max(inst_ready[iid].values())
            end = start + rec.get("est_us", 0.0)
            rec["sim_start_us"], rec["sim_end_us"] = round(start, 3), round(end, 3)
            for m in members:
                push(m, end)

    for nid, k in remaining.items():
        if k == 0:
            node_ready(nid, 0.0)
    done = 0
    while heap:
        t, nid = heapq.heappop(heap)
        if nid in finish:
            continue
        finish[nid] = t
        done += 1
        for s in succs[nid]:
            ready_t[s] = max(ready_t[s], t)
            remaining[s] -= 1
            if remaining[s] == 0:
                node_ready(s, ready_t[s])
    stuck = len(nodes) - done
    per_rank = collections.defaultdict(float)
    busy = collections.defaultdict(float)
    for nid, t in finish.items():
        n = nodes[nid]
        per_rank[n["rank"]] = max(per_rank[n["rank"]], t)
        if n["type"] == "COMP":
            busy[n["rank"]] += n["dur_us"]
    it = max(per_rank.values()) if per_rank else 0.0
    return {"iteration_us": round(it, 3), "unfinished_nodes": stuck,
            "per_rank_us": {str(k): round(v, 3) for k, v in sorted(per_rank.items())},
            "compute_busy_us": {str(k): round(v, 3) for k, v in sorted(busy.items())},
            "finish": finish}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Estimate comm times and replay a global DAG")
    ap.add_argument("global_dag")
    ap.add_argument("--cluster", help="JSON file overriding " + ", ".join(DEFAULT_CLUSTER))
    ap.add_argument("--gpus-per-node", type=int)
    ap.add_argument("--out", help="write annotated DAG here (default: <in>.sim.json)")
    a = ap.parse_args(argv)
    cluster = dict(DEFAULT_CLUSTER)
    if a.cluster:
        cluster.update(load_json(a.cluster))
    if a.gpus_per_node:
        cluster["gpus_per_node"] = a.gpus_per_node
    doc = load_json(a.global_dag)
    annotate(doc, cluster)
    res = replay(doc)
    for n in doc["nodes"]:
        if n["id"] in res["finish"]:
            n["sim_end_us"] = round(res["finish"][n["id"]], 3)
    res.pop("finish")
    doc["simulation"] = res
    out = a.out or a.global_dag.replace(".json", "") + ".sim.json"
    with open(out, "w") as f:
        json.dump(doc, f)
    comp = max(res["compute_busy_us"].values()) if res["compute_busy_us"] else 0
    print(f"cluster: {cluster}")
    print(f"estimated iteration time: {res['iteration_us']/1e3:.3f} ms "
          f"(max per-rank compute busy {comp/1e3:.3f} ms, "
          f"exposed comm+idle {(res['iteration_us']-comp)/1e3:.3f} ms)")
    if res["unfinished_nodes"]:
        print(f"WARNING: {res['unfinished_nodes']} nodes never became ready (cycle or unmatched comm)")
    tot = collections.Counter()
    for c in doc["collectives"]:
        tot[(c["coll"], ",".join(c.get("axes") or []))] += c["est_us"]
    for p in doc["p2p"]:
        tot[("P2P", "pp")] += p["est_us"]
    for k, v in tot.most_common():
        print(f"  sum est time {k[0]+'@'+k[1]:30s} {v/1e3:10.3f} ms (all instances)")
    print("->", out)


if __name__ == "__main__":
    main()
