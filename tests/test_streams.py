"""Unit test of the stream/event dependency model on a synthetic GPU trace that
mimics FSDP2: copy-in on a side stream, all-gather issued on the all-gather
stream, compute on the default stream overlapping with it, then the default
stream waits on the all-gather event before the next layer.

    python tests/test_streams.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dagtrace.extract import RankTrace  # noqa: E402


def op(i, parent, name, inputs=None):
    return {"id": i, "name": name, "ctrl_deps": parent,
            "inputs": {"values": inputs or [], "shapes": [], "types": ["Tensor(c10::BFloat16)",
                                                                       "Tensor(c10::BFloat16)"]},
            "outputs": {"values": [], "shapes": [], "types": []},
            "attrs": [{"name": "rf_id", "type": "uint64", "value": i},
                      {"name": "op_schema", "type": "string", "value": name + "(...)"}]}


def rf(i, parent, name):
    n = op(i, parent, name)
    n["attrs"][1]["value"] = ""
    return n


def kernel(ext, stream, dur, ts):
    return {"ph": "X", "cat": "kernel", "name": "k", "ts": ts, "dur": dur,
            "args": {"External id": ext, "stream": stream}}


def cpu(i, dur=1.0):
    return {"ph": "X", "cat": "cpu_op", "name": "x", "ts": 0, "dur": dur,
            "args": {"External id": 1000 + i, "Record function id": i}}


def build():
    T = lambda tid, sid, n: [tid, sid, 0, n, 2, "cuda:0"]  # noqa: E731
    nodes = [
        rf(1, 1, "[pytorch|profiler|execution_trace|process]"),
        rf(2, 1, "[pytorch|profiler|execution_trace|thread]"),
        rf(5, 2, "dagcalib::7"), op(6, 5, "aten::add_"),
        rf(8, 2, "dagcalib::20"), op(9, 8, "aten::add_"),
        rf(10, 2, "dagtrace::step#1"),
        op(11, 10, "aten::copy_"),                                   # copy-in, stream 20
        rf(12, 10, "dagevt::rec|e=1|s=20"),
        rf(13, 10, "dagevt::wait|e=1|s=21"),
        rf(14, 10, "dagcomm::all_gather_into_tensor#1|pg=5|peer=-|async=0|stream=21"),
        op(15, 14, "c10d::_allgather_base_", [T(100, 50, 2048), T(101, 51, 1024)]),
        rf(16, 10, "dagevt::rec|e=2|s=21"),
        op(17, 10, "aten::mm"),                                      # layer 0, stream 7
        rf(18, 10, "dagevt::wait|e=2|s=7"),
        op(19, 10, "aten::mm"),                                      # layer 1, stream 7
    ]
    kin = {"traceEvents": [cpu(i) for i in (6, 9, 11, 15, 17, 19)] + [
        kernel(1006, 7, 1, 0), kernel(1009, 20, 1, 0),
        kernel(1011, 20, 5.0, 10), kernel(1017, 7, 100.0, 20), kernel(1019, 7, 80.0, 130)]}
    meta = {"rank": 0, "world_size": 2, "mode": "fake",
            "topology": {"groups": {"5": {"ranks": [0, 1], "axes": ["dp_shard"], "partition": [[0, 1]]}}}}
    return {"schema": "test", "nodes": nodes}, kin, meta


def main():
    et, kin, meta = build()
    d = tempfile.mkdtemp()
    ep, kp = os.path.join(d, "et.json"), os.path.join(d, "k.json")
    json.dump(et, open(ep, "w"))
    json.dump(kin, open(kp, "w"))
    rt = RankTrace(ep, kp, meta).build(step_marker="dagtrace::step#1")
    fine = rt.fine()
    edges = {(a, b, k) for a, b, k in fine["edges"]}
    print("nodes:", [(n["id"], n.get("stream", n.get("issue_stream"))) for n in fine["nodes"]])
    print("edges:", sorted(edges))
    assert rt.calib == {"7": 7, "20": 20} and rt.calib_identity, rt.calib
    assert ("c11", "m0", "issue") in edges, "AG must wait for copy-in (event)"
    assert not any(b == "c17" and a == "m0" for a, b, _ in edges), "layer-0 compute must overlap the AG"
    assert ("m0", "c19", "complete") in edges, "layer-1 compute must wait for the AG (event)"
    assert ("c17", "c19", "stream") in edges
    ag = [n for n in fine["nodes"] if n["type"] == "COMM"][0]
    assert ag["bytes"] == 2048 * 2 and ag["issue_stream"] == 21 and ag["axes"] == ["dp_shard"]
    co = rt.coarse()
    assert len([n for n in co["nodes"] if n["type"] == "COMP"]) == 3
    print("OK: stream/event dependency model behaves as expected")


if __name__ == "__main__":
    main()
