"""Compare a DAG captured on ONE GPU with the fake backend against the DAG of
the same configuration captured in a REAL multi-GPU run (titan_capture --real).

This is the experiment that establishes trust in the single-GPU method:

  structure   identical sequence of collectives (type, group axis, bytes, peer),
              identical node sequence and identical dependency edges
  compute     per-segment and total compute time: single-GPU (isolated) vs
              real run (kernels share the GPU with NCCL -> interference)
  comm        measured NCCL kernel time per collective in the real run, and
              (optionally) the error of the alpha-beta estimate

Usage:
  python -m dagtrace.compare traces/fake_cfg traces/real_cfg [--cluster c.json]
"""
from __future__ import annotations

import collections
import difflib
import json
import os
import re

from dagtrace.extract import load_json


def _last_dag(rank_dir):
    steps = sorted(int(m.group(1)) for f in os.listdir(rank_dir)
                   if (m := re.match(r"step(\d+)\.dag\.json$", f)))
    if not steps:
        return None
    return load_json(os.path.join(rank_dir, f"step{steps[-1]}.dag.json"))


def _canon(dag):
    idx = {n["id"]: i for i, n in enumerate(dag["nodes"])}
    seq = []
    for n in dag["nodes"]:
        if n["type"] == "COMM":
            seq.append(("COMM", n["coll"], ",".join(n.get("axes") or []), n["bytes"], n.get("peer")))
        else:
            seq.append(("COMP",))
    edges = {(idx[a], idx[b], k) for a, b, k in dag["edges"] if a in idx and b in idx}
    return seq, edges


def compare_rank(fake, real, cluster=None):
    out = {}
    fs, fe = _canon(fake)
    rs, re_ = _canon(real)
    fcomm = [x for x in fs if x[0] == "COMM"]
    rcomm = [x for x in rs if x[0] == "COMM"]
    out["comm_sequence_identical"] = fcomm == rcomm
    out["n_comm"] = [len(fcomm), len(rcomm)]
    if fcomm != rcomm:
        sm = difflib.SequenceMatcher(a=fcomm, b=rcomm, autojunk=False)
        out["comm_sequence_similarity"] = round(sm.ratio(), 4)
        diffs = [op for op in sm.get_opcodes() if op[0] != "equal"][:5]
        out["comm_first_differences"] = [
            {"op": t, "fake": fcomm[i1:i2][:3], "real": rcomm[j1:j2][:3]} for t, i1, i2, j1, j2 in diffs]
    out["node_sequence_identical"] = fs == rs
    out["n_nodes"] = [len(fs), len(rs)]
    inter = len(fe & re_)
    out["edges_identical"] = fe == re_
    out["edge_jaccard"] = round(inter / max(1, len(fe | re_)), 4)

    fcomp = [n for n in fake["nodes"] if n["type"] == "COMP"]
    rcomp = [n for n in real["nodes"] if n["type"] == "COMP"]
    ft, rt = sum(n["dur_us"] for n in fcomp), sum(n["dur_us"] for n in rcomp)
    out["compute_total_us"] = {"fake": round(ft, 1), "real": round(rt, 1),
                               "rel_error": round((ft - rt) / rt, 4) if rt else None}
    by_phase = {}
    for ph in ("fwd", "bwd", "opt"):
        a = sum(n["dur_us"] for n in fcomp if n["phase"] == ph)
        b = sum(n["dur_us"] for n in rcomp if n["phase"] == ph)
        by_phase[ph] = {"fake": round(a, 1), "real": round(b, 1),
                        "rel_error": round((a - b) / b, 4) if b else None}
    out["compute_by_phase_us"] = by_phase
    if len(fcomp) == len(rcomp) and rt:
        errs = [(abs(a["dur_us"] - b["dur_us"]), b["dur_us"]) for a, b in zip(fcomp, rcomp)]
        out["compute_segment_wape"] = round(sum(e for e, _ in errs) / max(1e-9, sum(w for _, w in errs)), 4)
        big = sorted(zip(fcomp, rcomp), key=lambda p: -p[1]["dur_us"])[:10]
        out["largest_segments"] = [{"fake_us": a["dur_us"], "real_us": b["dur_us"],
                                    "top_op": (b.get("top_ops") or [["?"]])[0][0]} for a, b in big]
    rcm = [n for n in real["nodes"] if n["type"] == "COMM" and n.get("measured_us") is not None]
    if rcm:
        agg = collections.defaultdict(lambda: [0, 0.0, 0])
        for n in rcm:
            k = f"{n['coll']}@{','.join(n.get('axes') or ['?'])}"
            agg[k][0] += 1
            agg[k][1] += n["measured_us"]
            agg[k][2] += n["bytes"]
        out["real_comm_measured"] = {k: {"count": v[0], "total_us": round(v[1], 1),
                                         "bytes": v[2],
                                         "eff_GBps": round(v[2] / v[1] / 1e3, 2) if v[1] else None}
                                     for k, v in sorted(agg.items())}
        if cluster:
            from dagtrace.simulate import comm_time_us
            pairs = []
            for n in rcm:
                ranks = n.get("group_ranks") or []
                if n["coll"] in ("SEND", "RECV") and n.get("peer") is not None:
                    ranks = [real.get("rank", 0), n["peer"]]
                est = comm_time_us(n["coll"], n["bytes"], ranks, cluster)
                pairs.append((est, n["measured_us"]))
            m = sum(b for _, b in pairs)
            out["comm_model_wape"] = round(sum(abs(a - b) for a, b in pairs) / m, 4) if m else None
            out["comm_model_total_us"] = {"estimate": round(sum(a for a, _ in pairs), 1),
                                          "measured": round(m, 1)}
    return out


def compare(fake_dir, real_dir, cluster=None):
    res = {"fake": os.path.abspath(fake_dir), "real": os.path.abspath(real_dir), "ranks": {}}
    for d in sorted(os.listdir(fake_dir)):
        m = re.match(r"rank(\d+)$", d)
        if not m:
            continue
        rd = os.path.join(real_dir, d)
        if not os.path.isdir(rd):
            continue
        f, r = _last_dag(os.path.join(fake_dir, d)), _last_dag(rd)
        if f is None or r is None:
            continue
        res["ranks"][int(m.group(1))] = compare_rank(f, r, cluster)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fake_dir")
    ap.add_argument("real_dir")
    ap.add_argument("--cluster", help="cluster JSON (see dagtrace.simulate) to score the comm model")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cluster = None
    if a.cluster:
        from dagtrace.simulate import DEFAULT_CLUSTER
        cluster = dict(DEFAULT_CLUSTER)
        cluster.update(load_json(a.cluster))
    res = compare(a.fake_dir, a.real_dir, cluster)
    out = a.out or os.path.join(a.fake_dir, "compare_vs_real.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    if not res["ranks"]:
        print("no common rank directories with extracted DAGs")
        return 1
    for r, c in res["ranks"].items():
        print(f"rank {r}:")
        print(f"  collective sequence identical : {c['comm_sequence_identical']}  ({c['n_comm'][0]} vs {c['n_comm'][1]})")
        print(f"  node sequence identical       : {c['node_sequence_identical']}  ({c['n_nodes'][0]} vs {c['n_nodes'][1]})")
        print(f"  edges identical               : {c['edges_identical']}  (Jaccard {c['edge_jaccard']})")
        ct = c["compute_total_us"]
        print(f"  compute total fake/real       : {ct['fake']/1e3:.3f} / {ct['real']/1e3:.3f} ms "
              f"(rel. error {ct['rel_error']})")
        if "compute_segment_wape" in c:
            print(f"  per-segment compute WAPE      : {c['compute_segment_wape']}")
        if "comm_model_wape" in c:
            print(f"  comm model WAPE vs measured   : {c['comm_model_wape']}")
        for d in c.get("comm_first_differences", []):
            print("   diff:", d)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
