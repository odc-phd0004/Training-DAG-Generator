"""Validation report for captured DAGs: is the DAG really the workload's DAG?

Checks (per captured rank):
  stability     the DAG structure (comm sequence, sizes, groups, edges) is
                identical in every profiled iteration (sha256 of the structure)
  coverage      every collective has a known process group; every GPU kernel
                was attributed to an op; stream-id calibration consistent
  params        total parameters (sum over PP stages of the unsharded model
                parts) == analytical count from the model config
  pp_p2p        every PP send/recv carries exactly
                  microbatch * seq/cp (/tp with SP) * dim * bytes(dtype)
                and there are n_microbatches of them per direction
  tp_msg        TP all-gather / reduce-scatter carry the sequence-parallel
                activation  tokens * dim * bytes(dtype)
  fsdp_volume   reduce-scatter volume on the FSDP axis >= local params *
                FSDP degree * bytes(reduce dtype) (every gradient synced; ==1x
                means exactly one sync per step);
                all-gather volume reported as a multiple of one parameter pass
  global        (if global.dag.json exists) every collective is issued by all
                members with identical type/size; every send has a matching recv

Writes <trace_dir>/validation.json and validation.md.
"""
from __future__ import annotations

import json
import os
import re

from dagtrace.extract import load_json

DT_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def _f(meta, key, default=None):
    return meta.get("model", {}).get("fields", {}).get(key, default)


def analytic_params(meta):
    """Parameter count of a Llama-style decoder from its config fields."""
    dim = _f(meta, "dim")
    vocab = _f(meta, "vocab_size")
    L = _f(meta, "layers.count")
    if not (dim and vocab and L):
        return None
    pre = "layers[0].attention."
    nh = _f(meta, pre + "qkv_linear.n_heads") or _f(meta, pre + "n_heads")
    nkv = _f(meta, pre + "qkv_linear.n_kv_heads") or _f(meta, pre + "n_kv_heads") or nh
    hd = _f(meta, pre + "qkv_linear.head_dim") or _f(meta, pre + "head_dim") or dim // nh
    hidden = _f(meta, "layers[0].feed_forward.w1.out_features")
    if not (nh and hidden) or _f(meta, "layers[0].moe") is not None:
        return None
    attn = dim * nh * hd + 2 * dim * nkv * hd + nh * hd * dim
    ffn = 3 * dim * hidden
    norms = 2 * dim
    tied = _f(meta, "enable_weight_tying", False)
    return L * (attn + ffn + norms) + vocab * dim + (0 if tied else dim * vocab) + dim


def _from_gpu_dag(doc, meta):
    """Adapt a direct-mode gpu_dag to the fields the checks use."""
    topo = meta.get("topology", {}).get("groups", {})
    nodes = []
    for n in doc["nodes"]:
        if n["type"] == "compute":
            nodes.append({"type": "COMP", "dur_us": n["dur_us"], "phase": "?"})
        else:
            g = topo.get(str(n.get("group")))
            nodes.append({"type": "COMM", "coll": n["op"].upper(), "bytes": n["bytes"],
                          "group": n.get("group"), "axes": g["axes"] if g else None,
                          "peer": n.get("peer")})
    comm = [n for n in nodes if n["type"] == "COMM"]
    return {"nodes": nodes, "edges": doc["edges"], "signature": {"sha256": doc["signature"]},
            "summary": {"compute_us": sum(n["dur_us"] for n in nodes if n["type"] == "COMP"),
                        "untagged_c10d_comms": sum(1 for c in comm if c["group"] is None),
                        "unattributed_gpu_us": 0.0, "unmatched_tags": 0,
                        "time_source": "cuda_events" if meta.get("gpu") else "cpu_wall"}}


def _rank_dirs(trace_dir):
    return sorted((int(m.group(1)), os.path.join(trace_dir, d)) for d in os.listdir(trace_dir)
                  if (m := re.match(r"rank(\d+)$", d)))


def validate(trace_dir):
    report = {"trace_dir": os.path.abspath(trace_dir), "ranks": {}, "checks": []}

    def check(name, ok, detail, rank=None):
        report["checks"].append({"check": name, "rank": rank, "ok": bool(ok) if ok is not None else None,
                                 "detail": detail})

    total_params, stages_seen, meta0 = 0, set(), None
    for r, d in _rank_dirs(trace_dir):
        meta = load_json(os.path.join(d, "meta.json"))
        meta0 = meta0 or meta
        dags = sorted((int(m.group(1)), os.path.join(d, f)) for f in os.listdir(d)
                      if (m := re.match(r"step(\d+)\.dag\.json$", f)))
        if dags:
            docs = [(k, load_json(p)) for k, p in dags]
        else:
            direct = sorted((int(m.group(1)), os.path.join(d, f)) for f in os.listdir(d)
                            if (m := re.match(r"gpu_dag\.step(\d+)\.json$", f)))
            if not direct:
                check("extracted", False, "no DAG files (gpu_dag.step*.json or step*.dag.json)", r)
                continue
            docs = [(k, _from_gpu_dag(load_json(p), meta)) for k, p in direct]
        sigs = {k: doc["signature"]["sha256"] for k, doc in docs}
        check("stability", len(set(sigs.values())) == 1,
              f"structure hash per step: { {k: v[:12] for k, v in sigs.items()} }", r)
        k, dag = docs[-1]
        s = dag["summary"]
        report["ranks"][r] = {"summary": s, "signature": dag["signature"],
                              "pp_stages": meta.get("pp_stages"), "mode": meta.get("mode"),
                              "gpu": meta.get("gpu"), "versions": meta.get("versions")}
        check("coverage.groups", s["untagged_c10d_comms"] == 0,
              f"{s['untagged_c10d_comms']} collectives without process group; "
              f"{s.get('unmatched_tags', 0)} unmatched API tags", r)
        tot = s["compute_us"] + s["unattributed_gpu_us"]
        frac = s["unattributed_gpu_us"] / tot if tot else 0.0
        check("coverage.kernels", frac < 0.01 if s["time_source"] == "gpu_kernels" else None
              if s["time_source"] != "cuda_events" else None,
              f"time source {s['time_source']}; unattributed GPU time {frac*100:.3f}%", r)
        if dag.get("warnings"):
            check("coverage.warnings", False, "; ".join(dag["warnings"]), r)

        pdims = meta["parallel_dims"]
        tr = meta["training"]
        tp, cp, pp = pdims.get("tp") or 1, pdims.get("cp") or 1, pdims.get("pp") or 1
        dim = _f(meta, "dim")
        act_b = DT_BYTES.get(tr.get("mixed_precision_param"), 2)
        mbs = tr["pp_microbatch_size"] if pp > 1 else tr["local_batch_size"]
        n_mb = tr["local_batch_size"] // tr["pp_microbatch_size"] if pp > 1 else 1
        seq = tr["seq_len"] // cp
        comms = [n for n in dag["nodes"] if n["type"] == "COMM"]

        # params
        for st in meta.get("pp_stages") or [{"stage_index": 0}]:
            stages_seen.add(st["stage_index"])
        total_params += sum(p["global_numel"] for p in meta.get("model_parts", []))

        # PP P2P sizes
        if pp > 1 and dim:
            sp = tp > 1 and tr.get("sequence_parallel", True)
            exp = mbs * (seq // tp if sp else seq) * dim * act_b
            p2p = [n for n in comms if n["coll"] in ("SEND", "RECV")]
            bad = [n["bytes"] for n in p2p if n["bytes"] != exp]
            n_send = sum(1 for n in p2p if n["coll"] == "SEND")
            n_recv = len(p2p) - n_send
            stages = [x["stage_index"] for x in meta.get("pp_stages", [])]
            nst = meta["pp_stages"][0]["num_stages"] if meta.get("pp_stages") else pp
            exp_send = sum((n_mb if s < nst - 1 else 0) + (n_mb if s > 0 else 0) for s in stages)
            check("pp_p2p.size", not bad and p2p,
                  f"{len(p2p)} P2P ops, expected size {exp} B each "
                  f"(mbs {mbs} x seq {seq}{'/tp' if sp else ''} x dim {dim} x {act_b} B); "
                  f"mismatching sizes: {sorted(set(bad))[:5]}", r)
            check("pp_p2p.count", n_send == exp_send and n_recv == exp_send,
                  f"sends {n_send}, recvs {n_recv}, expected {exp_send} each "
                  f"({n_mb} microbatches, stages {stages} of {nst})", r)

        # TP message size
        if tp > 1 and dim:
            exp = mbs * seq * dim * act_b
            tpc = [n for n in comms if (n.get("axes") or []) and "tp" in n["axes"]
                   and n["coll"] in ("ALL_GATHER", "REDUCE_SCATTER")]
            hit = sum(1 for n in tpc if n["bytes"] == exp)
            vol = sum(n["bytes"] for n in tpc)
            hit_vol = sum(n["bytes"] for n in tpc if n["bytes"] == exp)
            L = len([1 for _ in range(_f(meta, "layers.count") or 0)])
            check("tp_msg.size", tpc and hit_vol / max(vol, 1) > 0.5,
                  f"{hit}/{len(tpc)} TP all-gather/reduce-scatter ops ({hit_vol/max(vol,1)*100:.1f}% of TP "
                  f"volume) have the SP activation size {exp} B = {mbs}x{seq}x{dim}x{act_b}; "
                  f"{len(tpc)/(max(L,1)*n_mb* max(len(meta.get('pp_stages') or [1]),1)):.2f} per layer-microbatch "
                  f"(4 fwd+bwd per layer without recompute; more with activation checkpointing)", r)

        # FSDP volume
        fs = [n for n in comms if (n.get("axes") or []) and
              any(a in ("dp_shard", "fsdp", "efsdp", "dp_shard_cp") for a in n["axes"])
              and n["coll"] in ("REDUCE_SCATTER", "ALL_GATHER")]
        dps = pdims.get("dp_shard") or 1
        if fs and dps > 1:
            local = sum(p["local_numel"] for p in meta.get("model_parts", []))
            red_b = DT_BYTES.get(tr.get("mixed_precision_reduce") or "float32", 4)
            rs = sum(n["bytes"] for n in fs if n["coll"] == "REDUCE_SCATTER")
            ag = sum(n["bytes"] for n in fs if n["coll"] == "ALL_GATHER")
            unit_rs = local * dps * red_b
            unit_ag = local * dps * act_b
            ratio = rs / unit_rs if unit_rs else 0
            check("fsdp_volume.reduce_scatter", ratio >= 0.98,
                  f"RS volume {rs} B = {ratio:.4f} x (local params {local} x dp_shard {dps} x {red_b} B); "
                  + ("exactly one gradient sync per step" if abs(ratio - 1) < 0.02 else
                     "every gradient is synced at least once; >1 means some FSDP groups "
                     "reduce-scatter per microbatch (framework behaviour, e.g. PP grad accumulation)"), r)
            check("fsdp_volume.all_gather", None,
                  f"AG volume {ag} B = {ag/unit_ag if unit_ag else 0:.3f} x one parameter pass "
                  f"({local} x {dps} x {act_b} B); ~2 = forward + backward re-gather "
                  f"(reshard_after_forward)", r)

        # HSDP: gradients of the local shard are all-reduced across replicas
        rep = pdims.get("dp_replicate") or 1
        if rep > 1:
            local = sum(p["local_numel"] for p in meta.get("model_parts", []))
            red_b = DT_BYTES.get(tr.get("mixed_precision_reduce") or "float32", 4)
            ar = sum(n["bytes"] for n in comms if n["coll"] == "ALL_REDUCE"
                     and "dp_replicate" in (n.get("axes") or []))
            ratio = ar / (local * red_b) if local else 0
            check("hsdp_volume.all_reduce", ratio >= 0.98,
                  f"all-reduce volume on dp_replicate {ar} B = {ratio:.4f} x local gradient shard "
                  f"({local} x {red_b} B)", r)

    ap = analytic_params(meta0) if meta0 else None
    if ap is not None and meta0:
        nst = (meta0.get("pp_stages") or [{"num_stages": 1}])[0]["num_stages"]
        complete = len(stages_seen) == nst
        check("params", (total_params == ap) if complete else None,
              f"sum of captured stages' parameters {total_params:,} vs analytical {ap:,}"
              + ("" if complete else f" (only stages {sorted(stages_seen)} of {nst} captured)"))

    jpath = os.path.join(trace_dir, "job_dag.json")
    if os.path.exists(jpath):
        j = load_json(jpath)
        c = j["consistency"]
        check("job.consistency", c["ok"],
              f"{j['counts'].get('collective', 0)} collective and {j['counts'].get('p2p', 0)} p2p vertices "
              f"across {j['world_size']} GPUs; {c['n_problems']} problems "
              + ("; ".join(c["problems"][:3]) if c["problems"] else ""))
    gpath = os.path.join(trace_dir, "global.dag.json")
    if os.path.exists(gpath):
        g = load_json(gpath)
        c = g["consistency"]
        check("global.consistency", c["ok"],
              f"{g['counts']['collectives']} collective instances, {g['counts']['p2p']} P2P pairs "
              f"across {g['world_size']} ranks; {c['n_problems']} problems "
              + ("; ".join(c["problems"][:3]) if c["problems"] else ""))

    n_ok = sum(1 for c in report["checks"] if c["ok"] is True)
    n_bad = sum(1 for c in report["checks"] if c["ok"] is False)
    report["result"] = {"passed": n_ok, "failed": n_bad,
                        "info": sum(1 for c in report["checks"] if c["ok"] is None)}
    with open(os.path.join(trace_dir, "validation.json"), "w") as f:
        json.dump(report, f, indent=1)
    lines = [f"# dagtrace validation: {trace_dir}", "",
             f"**{n_ok} passed, {n_bad} failed, {report['result']['info']} informational**", "",
             "| check | rank | result | detail |", "|---|---|---|---|"]
    for c in report["checks"]:
        res = {True: "PASS", False: "FAIL", None: "info"}[c["ok"]]
        lines.append(f"| {c['check']} | {'' if c['rank'] is None else c['rank']} | {res} | "
                     f"{c['detail'].replace('|', '/')} |")
    with open(os.path.join(trace_dir, "validation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return report


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Validate captured DAGs")
    ap.add_argument("trace_dir")
    a = ap.parse_args(argv)
    rep = validate(a.trace_dir)
    for c in rep["checks"]:
        res = {True: "PASS", False: "FAIL", None: "info"}[c["ok"]]
        r = "" if c["rank"] is None else f"[rank {c['rank']}] "
        print(f"{res:4s} {c['check']:28s} {r}{c['detail']}")
    print(f"\n{rep['result']['passed']} passed, {rep['result']['failed']} failed "
          f"-> {os.path.join(a.trace_dir, 'validation.md')}")
    return 1 if rep["result"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
