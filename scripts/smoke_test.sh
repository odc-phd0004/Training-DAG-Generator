#!/usr/bin/env bash
# End-to-end check that everything works on this machine (~1-2 min on a GPU):
# TorchTitan's tiny debug model as an 8-GPU job with PP=2 x TP=2 x FSDP=2,
# captured on ONE GPU as per-GPU DAGs, assembled into the job DAG and validated.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
python tests/test_streams.py
python tests/test_online.py
OUT_ROOT="${OUT_ROOT:-$HERE/traces}" bash scripts/capture_job.sh smoke_debug_pp2_tp2_fsdp2 8 \
  --warmup 2 --profile-steps 2 --all-ranks -- \
  --module llama3 --config llama3_debugmodel \
  --training.seq_len 2048 --training.local_batch_size 8 \
  --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 \
  --parallelism.pipeline_parallel_schedule 1F1B --parallelism.pipeline_parallel_microbatch_size 2
python - "${OUT_ROOT:-$HERE/traces}/smoke_debug_pp2_tp2_fsdp2" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1] + "/validation.json"))
bad = [c for c in v["checks"] if c["ok"] is False]
for c in bad:
    print("FAILED:", c["check"], c.get("rank"), c["detail"])
print(f"\nsmoke test: {v['result']['passed']} checks passed, {v['result']['failed']} failed")
sys.exit(1 if bad else 0)
EOF
