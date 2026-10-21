#!/usr/bin/env bash
# One-time environment setup for dagtrace on a single-GPU Linux machine.
#
#   bash setup.sh                 # creates ./.venv and ./third_party/
#   VENV=/path/to/venv bash setup.sh
#
# Pinned versions (reproducibility matters for a paper artifact):
#   PyTorch 2.14.1 (CUDA 13.0 wheels from PyPI), TorchTitan v0.3.0,
#   Chakra (MLCommons) at a fixed commit.
#
# If your NVIDIA driver is older than R580 (CUDA 13 needs >= 580), use the
# CUDA 12.8 build instead:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
PY="${PYTHON:-python3}"
TT_REF="v0.3.0"
CHAKRA_REF="9ff3e3e2f276b4c0554a83f8747bf00b2786fa85"
TORCH_INDEX="${TORCH_INDEX:-}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "Checking system"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (NVIDIA driver missing?)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [[ -z "$TORCH_INDEX" && "$DRIVER_MAJOR" -lt 580 ]]; then
  echo "Driver $DRIVER_MAJOR < 580: switching to CUDA 12.8 PyTorch wheels."
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi
"$PY" - <<'EOF' || die "Python >= 3.11 is required (TorchTitan requirement). Set PYTHON=/path/to/python3.11+"
import sys; assert sys.version_info >= (3, 11), sys.version
print("python", sys.version.split()[0])
EOF
command -v git >/dev/null || die "git is required"

say "Creating virtualenv at $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

say "Installing PyTorch 2.14.1 ${TORCH_INDEX:+from $TORCH_INDEX}"
if [[ -n "$TORCH_INDEX" ]]; then
  pip install "torch==2.14.1" --index-url "$TORCH_INDEX"
else
  pip install "torch==2.14.1"
fi

say "Fetching TorchTitan $TT_REF"
mkdir -p "$HERE/third_party"
if [[ ! -d "$HERE/third_party/torchtitan/.git" ]]; then
  git clone --quiet --depth 1 --branch "$TT_REF" https://github.com/pytorch/torchtitan.git "$HERE/third_party/torchtitan"
fi
pip install -q -r "$HERE/third_party/torchtitan/requirements.txt" -c "$HERE/constraints.txt"
pip install -q -e "$HERE/third_party/torchtitan" --no-deps

say "Installing Chakra @ ${CHAKRA_REF:0:10} (for ASTRA-sim export)"
pip install -q "chakra @ git+https://github.com/mlcommons/chakra.git@$CHAKRA_REF" -c "$HERE/constraints.txt"

say "Installing dagtrace"
pip install -q orjson || true
pip install -q -e "$HERE" --no-deps

say "Sanity checks"
python - <<'EOF'
import ctypes, torch
assert torch.__version__.startswith("2.14.1"), torch.__version__
assert torch.cuda.is_available(), "CUDA not available to PyTorch"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} cuda {torch.version.cuda} | {p.name} {p.total_memory/2**30:.0f} GiB "
      f"sm_{p.major}{p.minor}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()
import torchtitan, chakra  # noqa
from chakra.schema.protobuf.et_def_pb2 import Node  # noqa
from dagtrace import commtag
s = torch.cuda.Stream()
print("stream ids (driver):", commtag.stream_key(), commtag.stream_key(s))
assert commtag.stream_key().isdigit(), "cuStreamGetId unavailable -> stream mapping falls back to calibration"
print("OK")
EOF

cat <<EOF

Setup complete.  Activate with:
    source "$VENV/bin/activate"
Then run the smoke test (about 1-2 minutes):
    bash "$HERE/scripts/smoke_test.sh"
EOF
