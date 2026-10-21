"""Dump the process-group topology of the running job.

For every process group this rank belongs to we record its global ranks.  For
groups that come from a DeviceMesh we additionally record the mesh axis name
and the *complete partition* of the world into groups of that axis (via the
mesh's CuTe-style layout).  The partition is what lets the stitcher build the
DAG of ranks that were never executed: "rank x's TP group" is simply the block
of the TP partition that contains x.
"""
from __future__ import annotations

import gc

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def _mesh_axis_entries(mesh):
    """Yield (axis_name, 1-D submesh) for every dim of a DeviceMesh."""
    names = mesh.mesh_dim_names or tuple(f"dim{i}" for i in range(mesh.ndim))
    if mesh.ndim == 1:
        yield names[0], mesh
        return
    for i, n in enumerate(names):
        try:
            yield n, mesh[n] if mesh.mesh_dim_names else None
        except Exception:
            continue


def _partition(sub) -> list[list[int]] | None:
    try:
        t = sub._layout.remap_to_tensor(sub._rank_map)
        return t.reshape(-1, sub.size()).tolist()
    except Exception:
        return None


def collect(extra_meshes=()) -> dict:
    from torch.distributed.device_mesh import DeviceMesh

    groups: dict[str, dict] = {}
    for pg in list(c10d._world.pg_map.keys()):
        try:
            name = str(pg.group_name)
            groups[name] = {
                "ranks": dist.get_process_group_ranks(pg),
                "backend": str(dist.get_backend(pg)),
                "axes": [],
                "partition": None,
            }
        except Exception:
            continue

    meshes = list(extra_meshes)
    for o in gc.get_objects():
        try:
            if isinstance(o, DeviceMesh):
                meshes.append(o)
        except ReferenceError:      # dead weakproxy
            continue
    seen = set()
    for mesh in meshes:
        if id(mesh) in seen or mesh is None:
            continue
        seen.add(id(mesh))
        for axis, sub in _mesh_axis_entries(mesh):
            if sub is None:
                continue
            try:
                g = sub.get_group()
            except Exception:
                continue
            name = str(g.group_name)
            ent = groups.setdefault(name, {
                "ranks": dist.get_process_group_ranks(g),
                "backend": str(dist.get_backend(g)), "axes": [], "partition": None})
            if axis not in ent["axes"]:
                ent["axes"].append(axis)
            if ent["partition"] is None:
                ent["partition"] = _partition(sub)

    default = c10d._get_default_group()
    w = dist.get_world_size()
    dname = str(default.group_name)
    if dname in groups:
        groups[dname]["axes"] = groups[dname]["axes"] or ["world"]
        groups[dname]["partition"] = [list(range(w))]
    return {
        "world_size": w,
        "rank": dist.get_rank(),
        "default_group": dname,
        "groups": groups,
    }
