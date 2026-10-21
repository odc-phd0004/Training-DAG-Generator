"""Export a global DAG to Chakra execution traces (one .et per rank) + the
communicator-group file, the input format of ASTRA-sim.

  python -m dagtrace.to_chakra traces/X/global.dag.json --out chakra/X

writes chakra/X/et.<rank>.et and chakra/X/comm_group.json.

Node mapping:
  COMP          -> COMP_NODE      duration_micros = measured GPU time
  collective    -> COMM_COLL_NODE comm_type, comm_size (bytes), pg_name
  SEND / RECV   -> COMM_SEND_NODE / COMM_RECV_NODE  comm_src, comm_dst, comm_tag, comm_size
  dependencies  -> data_deps
"""
from __future__ import annotations

import collections
import json
import os

from dagtrace.extract import load_json


def export(global_dag, out_dir, prefix="et"):
    from chakra.schema.protobuf.et_def_pb2 import (ALL_GATHER, ALL_REDUCE, ALL_TO_ALL, BARRIER,
                                                   BROADCAST, COMM_COLL_NODE, COMM_RECV_NODE,
                                                   COMM_SEND_NODE, COMP_NODE, GATHER, REDUCE,
                                                   REDUCE_SCATTER, SCATTER)
    from chakra.schema.protobuf.et_def_pb2 import AttributeProto as Attr
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
    from chakra.src.third_party.utils.protolib import encodeMessage

    coll_enum = {"ALL_REDUCE": ALL_REDUCE, "ALL_GATHER": ALL_GATHER,
                 "REDUCE_SCATTER": REDUCE_SCATTER, "ALL_TO_ALL": ALL_TO_ALL,
                 "BROADCAST": BROADCAST, "REDUCE": REDUCE, "GATHER": GATHER,
                 "SCATTER": SCATTER, "BARRIER": BARRIER}
    doc = load_json(global_dag) if isinstance(global_dag, str) else global_dag
    os.makedirs(out_dir, exist_ok=True)

    pg_ids = {}

    def pg_name(ranks):
        key = tuple(sorted(ranks))
        if key not in pg_ids:
            pg_ids[key] = str(len(pg_ids) + 1)
        return pg_ids[key]

    p2p_tag = {p["id"]: i for i, p in enumerate(doc["p2p"])}
    by_rank = collections.defaultdict(list)
    for n in doc["nodes"]:
        by_rank[n["rank"]].append(n)
    preds = collections.defaultdict(list)
    for a, b, _ in doc["edges"]:
        preds[b].append(a)

    written = []
    for r, nodes in sorted(by_rank.items()):
        ids = {n["id"]: i for i, n in enumerate(nodes)}
        path = os.path.join(out_dir, f"{prefix}.{r}.et")
        with open(path, "wb") as f:
            encodeMessage(f, GlobalMetadata(attr=[
                Attr(name="schema", string_val="1.1.1-chakra.0.0.4"),
                Attr(name="source", string_val="dagtrace"),
                Attr(name="world_size", int64_val=doc["world_size"])]))
            for n in nodes:
                pb = Node(id=ids[n["id"]], name=n.get("name") or n.get("op") or n["id"])
                for p in preds[n["id"]]:
                    if p in ids:
                        pb.data_deps.append(ids[p])
                attrs = [Attr(name="is_cpu_op", bool_val=False)]
                if n["type"] == "COMP":
                    pb.type = COMP_NODE
                    pb.name = f"{n['phase']}:{n['id'].split(':', 1)[1]}"
                    pb.duration_micros = int(round(n["dur_us"]))
                    attrs.append(Attr(name="num_ops", int64_val=int(n.get("n_ops", 1))))
                elif n["coll"] in ("SEND", "RECV"):
                    pb.type = COMM_SEND_NODE if n["coll"] == "SEND" else COMM_RECV_NODE
                    src, dst = (r, n["peer"]) if n["coll"] == "SEND" else (n["peer"], r)
                    attrs += [Attr(name="comm_src", int32_val=int(src)),
                              Attr(name="comm_dst", int32_val=int(dst)),
                              Attr(name="comm_tag", int32_val=p2p_tag.get(n.get("instance"), 0)),
                              Attr(name="comm_size", int64_val=int(n["bytes"]))]
                else:
                    pb.type = COMM_COLL_NODE
                    attrs += [Attr(name="comm_type", int64_val=coll_enum.get(n["coll"], ALL_REDUCE)),
                              Attr(name="comm_size", int64_val=int(n["bytes"])),
                              Attr(name="pg_name", string_val=pg_name(n["group_ranks"]))]
                pb.attr.extend(attrs)
                encodeMessage(f, pb)
        written.append(path)
    groups = {v: list(k) for k, v in pg_ids.items()}
    with open(os.path.join(out_dir, "comm_group.json"), "w") as f:
        json.dump(groups, f, indent=1)
    return written, groups


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("global_dag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files, groups = export(a.global_dag, a.out)
    print(f"wrote {len(files)} Chakra traces and {len(groups)} communicator groups to {a.out}")


if __name__ == "__main__":
    main()
