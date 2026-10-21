"""Unit test of the direct (online) tracer's dependency model, CPU only.
FSDP2-like pattern: copy-in on stream 20, all-gather on stream 21 overlapping
compute on the main stream, main stream waits on the all-gather event.

    python tests/test_online.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dagtrace.online import Tracer  # noqa: E402


CLOCK = {}


def work(stream, ms=3):
    """Advance one stream's clock, like kernels running on that CUDA stream."""
    CLOCK[stream] = CLOCK.get(stream, 0.0) + ms / 1e3


def main():
    T = Tracer()
    T.cuda = False
    T._stamp = lambda s, obj=None: CLOCK.get(s, 0.0)   # per-stream time, like CUDA events
    s20, s21 = types.SimpleNamespace(cuda_stream=20), types.SimpleNamespace(cuda_stream=21)
    T.begin()                          # main-stream segment A
    work("main")                       # A: previous layer compute
    T.event_record(1, None)            # main records "params ready"
    T.event_wait(1, s20)               # copy-in stream waits on it
    work(20)                           # B: copy-in (stream 20)
    T.event_record(2, s20)
    T.event_wait(2, s21)
    T.comm("all_gather", 4096, [0, 1], "5", None, False, stream_obj=s21)
    T.event_record(3, s21)
    work("main")                       # C: layer-0 compute on main, overlaps the all-gather
    T.event_wait(3, None)              # main waits for the all-gather
    work("main")                       # D: layer-1 compute
    dag = T.end(min_us=500)
    nodes, edges = dag["nodes"], set(map(tuple, dag["edges"]))
    print("nodes:", [(n["id"], n["type"], n.get("op", ""), round(n.get("dur_us", 0) / 1e3, 1)) for n in nodes])
    print("edges:", sorted(edges))
    comp = {n["id"] for n in nodes if n["type"] == "compute"}
    agv = [n["id"] for n in nodes if n["type"] == "collective"][0]
    pred = lambda v: {u for u, w in edges if w == v}  # noqa: E731
    assert len(comp) == 4, comp
    (B,) = pred(agv) & comp                       # copy-in feeds the all-gather
    (A,) = pred(B)                                # copy-in waited on main's event
    (D,) = {v for v in comp if agv in pred(v)}    # the layer that consumes the all-gather
    (C,) = comp - {A, B, D}                       # the overlapped layer
    assert pred(C) == {A}, "layer-0 compute must not wait for the all-gather"
    assert pred(D) == {agv, C}, "layer-1 waits on all-gather and layer-0"
    print("OK: online tracer dependency model behaves as expected")


if __name__ == "__main__":
    main()
