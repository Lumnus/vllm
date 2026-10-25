#!/bin/bash
# B70-QSA-T: rebuild libb70_step_trace.so (Niluthon e536e33 kernels/b70-step-trace, Apache-2.0) for the 0.32 line:
# icpx 2026.1 (= intel-sycl-rt 2026.1.1 in t214-line032pr2) against torch 2.14.0+xpu. The perf-ab build
# (/work/engines/flash-next-perfc/lib, icpx 2026.0 vs opt-k4 torch) is the wrong ABI for this venv.
# Imports torch, so the GPUs must be hidden:
#   unshare -m bash -c "mount -t tmpfs none /dev/dri; bash -l <this script>"
set -euo pipefail
export CRUSHER_MODE=nogpu; unset LD_PRELOAD
VENV=${VENV:-/work/venvs/t214-line032pr2}
S=${S:-/work/engines/flash-next-perfc/src-niluthon-e536e33/kernels/b70-step-trace}
OUT=${OUT:-/work/cells/qsa-triton-C/lib}
mkdir -p "$OUT"
set +u; source /opt/intel/oneapi/compiler/2026.1/env/vars.sh >/dev/null 2>&1 || true; set -u
icpx --version | head -1
export LD_LIBRARY_PATH=$VENV/lib:${LD_LIBRARY_PATH:-}
T=$($VENV/bin/python -I -c 'import torch, os; print(os.path.dirname(torch.__file__))')
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device bmg' \
  -O2 -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ $S/b70_step_trace.sycl -o $OUT/libb70_step_trace.so \
  -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib -Wl,-rpath,$VENV/lib
sha256sum $OUT/libb70_step_trace.so
ldd $OUT/libb70_step_trace.so | grep -E "sycl|torch|c10|not found"
# Must show exactly one libsycl (the venv's). Card test, engine stopped, one card:
#   ZE_AFFINITY_MASK=1 $VENV/bin/python $S/test_step_trace.py $OUT/libb70_step_trace.so
