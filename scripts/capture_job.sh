#!/usr/bin/env bash
# Capture the compute/collective/p2p DAG of one training-job configuration on a
# single GPU, then build the job DAG and validate it.
#
#   bash scripts/capture_job.sh <name> <logical_ngpu> [options] -- <torchtitan args>
#
# Example (Llama-3 8B, 16 GPUs, TP=2 x FSDP=8):
#   bash scripts/capture_job.sh llama3_8b_tp2_fsdp8 16 -- \
#       --module llama3 --config llama3_8b --parallelism.tensor_parallel_degree 2
#
# Options (before --):
#   --layers K          truncate the model depth (memory)
#   --warmup N          un-traced warmup steps (default 3)
#   --profile-steps N   traced steps (default 2; >=2 enables the stability check)
#   --max-expand N      write explicit job_dag.json only up to N GPUs (default 1024)
#   --rank R            also write the per-GPU DAG of rank R (repeatable)
#   --all-ranks         write per_gpu/gpu_dag.rank<x>.json for every rank
# MODE=raw bash scripts/capture_job.sh ...  -> profiler-based path (ET + Kineto),
#   which additionally produces op-level DAGs, simulation and Chakra export.
# Output: traces/<name>/  (override with OUT_ROOT=...)
set -euo pipefail
NAME=${1:?name}; NGPU=${2:?logical number of GPUs}; shift 2
CAP=(); JOB=(); GPN=8
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --max-expand) JOB+=(--max-expand "$2"); shift 2 ;;
    --rank)       JOB+=(--rank "$2"); shift 2 ;;
    --all-ranks)  JOB+=(--all-ranks); shift ;;
    --gpus-per-node) GPN="$2"; shift 2 ;;
    *) CAP+=("$1"); shift ;;
  esac
done
[[ "${1:-}" == "--" ]] && shift
TT=("$@")
MODE="${MODE:-direct}"
OUT="${OUT_ROOT:-traces}/$NAME"
EXTRA=(--mode "$MODE")
[[ "${DAGTRACE_CPU_TEST:-0}" == "1" ]] && EXTRA+=(--cpu-test)
mkdir -p "$OUT"
LOG="$OUT/capture.log"
: > "$LOG"

run_stage() {
  local s=$1
  echo "[capture] $NAME: pipeline stage $s of a $NGPU-GPU job ..." | tee -a "$LOG"
  python -m dagtrace.titan_capture "${EXTRA[@]}" --out "$OUT" --ngpu "$NGPU" --pp-rank "$s" \
      "${CAP[@]}" -- "${TT[@]}" >> "$LOG" 2>&1 || {
    echo "capture failed for stage $s; last lines of $LOG:" >&2; tail -40 "$LOG" >&2; exit 1; }
  grep "\[dagtrace\]" "$LOG" | tail -n 2
}

run_stage 0
PP=$(python - "$OUT" <<'EOF'
import json, os, sys, re
d = sys.argv[1]
m = [json.load(open(os.path.join(d, x, "meta.json"))) for x in os.listdir(d)
     if re.match(r"rank\d+$", x) and os.path.exists(os.path.join(d, x, "meta.json"))]
print(max(x["parallel_dims"]["pp"] or 1 for x in m))
EOF
)
for ((s=1; s<PP; s++)); do run_stage "$s"; done

if [[ "$MODE" == "direct" ]]; then
  echo "[job dag]"; python -m dagtrace.jobdag "$OUT" "${JOB[@]}" | tee -a "$LOG"
  echo "[validate]"; python -m dagtrace.validate "$OUT" | tee -a "$LOG" || true
else
  echo "[extract]"; python -m dagtrace.extract "$OUT" | tee -a "$LOG"
  echo "[stitch]";  python -m dagtrace.stitch "$OUT" | tee -a "$LOG"
  echo "[validate]"; python -m dagtrace.validate "$OUT" | tee -a "$LOG" || true
  echo "[simulate]"; python -m dagtrace.simulate "$OUT/global.dag.json" --gpus-per-node "$GPN" | tee -a "$LOG"
  if python -c "import chakra" 2>/dev/null; then
    echo "[chakra]"; python -m dagtrace.to_chakra "$OUT/global.dag.json" --out "$OUT/chakra" | tee -a "$LOG"
  fi
fi
echo "done: $OUT"
