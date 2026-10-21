"""CPU shim used ONLY for CI-style testing of dagtrace on a machine without a GPU.

TorchTitan assumes an accelerator (it defaults its device type to "cuda").
This shim makes it pick "cpu" and fills in the few torch.cuda-style helpers it
calls on its device module.  Never import this on the GPU box.
"""
import types

import torch
import torch._utils as _tu

_tu._get_available_device_type = lambda: "cpu"

_cpu = torch.cpu
_defaults = {
    "memory_stats": lambda *a, **k: {},
    "reset_peak_memory_stats": lambda *a, **k: None,
    "empty_cache": lambda *a, **k: None,
    "max_memory_allocated": lambda *a, **k: 0,
    "max_memory_reserved": lambda *a, **k: 0,
    "memory_allocated": lambda *a, **k: 0,
    "memory_reserved": lambda *a, **k: 0,
    "get_device_name": lambda *a, **k: "cpu",
    "get_device_properties": lambda *a, **k: types.SimpleNamespace(
        name="cpu", total_memory=8 * 2**30, major=0, minor=0, multi_processor_count=1),
    "get_device_capability": lambda *a, **k: (0, 0),
    "mem_get_info": lambda *a, **k: (8 * 2**30, 8 * 2**30),
}
for k, v in _defaults.items():
    if not hasattr(_cpu, k):
        setattr(_cpu, k, v)
if not hasattr(_cpu, "memory"):
    _cpu.memory = types.SimpleNamespace(
        _record_memory_history=lambda *a, **k: None,
        _snapshot=lambda *a, **k: {},
        _dump_snapshot=lambda *a, **k: None,
    )
_orig_set_device = getattr(_cpu, "set_device", None)
_cpu.set_device = lambda *a, **k: None
_cpu.current_device = lambda: 0


def _patch_flex_for_cpu():
    """FlexAttention has no CPU backward; substitute causal SDPA (test only)."""
    import torch.nn.functional as F
    from torchtitan.models.common import attention as A

    def cpu_flex(q, k, v, *, score_mod=None, block_mask=None, scale=None,
                 enable_gqa=False, return_aux=None, kernel_options=None):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale,
                                             enable_gqa=enable_gqa)
        return out, types.SimpleNamespace(lse=None)

    A.FlexAttention.compiled_flex_attn = staticmethod(cpu_flex)


try:
    _patch_flex_for_cpu()
except Exception as e:  # torchtitan not importable yet
    print("cpu_shim: flex patch skipped:", e)
