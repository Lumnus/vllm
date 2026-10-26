#!/usr/bin/env bash
# B70-QSA-D06 (run runs/2026-W43/2026-10-25-qsa-triton-xpu, artifacts/D-qsa-levers.md): engine arms for the top-k
# levers on the S1 def (attn2 + native_det), the pattern of mkdefs-qsa-tune.sh. Every arm runs the levers tree with
# its own caches; the switches are env only. Refuses to overwrite unless OVERWRITE=1. Never starts anything.
#   usage: mkdefs-qsa-levers.sh [all|<suffix>...]
set -euo pipefail
E=${E:-/etc/crusher/engines}
B=${BASE:-qsa-dev-intel-autoround-v032-dflash7-s8-kv128-bt4096-rcram-expplug-attn2-ndet}
TREE=${TREE:-/work/src/vllm-v032-qsa-levers}
C=${CELL:-/work/cells/qsa-levers-D}
for f in $B.env $B.cmd $B.args; do [ -e "$E/$f" ] || { echo "missing base $E/$f" >&2; exit 2; }; done
put() { local f=$E/$1; if [ -e "$f" ] && [ "${OVERWRITE:-0}" != 1 ]; then echo "exists: $f (OVERWRITE=1)" >&2; exit 2; fi
        cat > "$f"; echo "wrote $f"; }
# mk <suffix> <description> <cmd filter> <env lines...>
mk() {
  local N=$B-$1 change=$2 cmdf=$3; shift 3
  { echo "# $N - Cell D 2026-10-26: = $B on the levers tree ($TREE, own caches) + $change"
    sed -e "s#^PYTHONPATH=.*#PYTHONPATH=$TREE#" \
        -e "s#^TRITON_CACHE_DIR=.*#TRITON_CACHE_DIR=$C/triton-cache#" \
        -e "s#^VLLM_CACHE_ROOT=.*#VLLM_CACHE_ROOT=$C/vllm-cache#" < $E/$B.env
    for l in "$@"; do echo "$l"; done; } | put $N.env
  "$cmdf" < $E/$B.cmd | put $N.cmd
  { echo "# $N - .args identical to $B.args"; cat $E/$B.args; } | put $N.args
}
same() { cat; }
prof_cmd() { sed "s#\$# --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$C/prof --profiler-config.torch_profiler_with_stack=false --profiler-config.max_iterations=40#"; }
LEGACY=(B70_QSA_TOPK_DET_IMPL=legacy B70_QSA_SCORE_DEFAULTS=legacy)
run() {
case "$1" in
lv-base)   mk lv-base   'D01/D03 switched off (legacy repair + GB300 scorer configs): must equal S1' same "${LEGACY[@]}" ;;
lv-d01)    mk lv-d01    'D01 select as torch ops (scorer configs legacy)' same B70_QSA_SCORE_DEFAULTS=legacy ;;
lv-d01t)   mk lv-d01t   'D01 select as the fused Triton kernels (scorer configs legacy)' same \
                        B70_QSA_SCORE_DEFAULTS=legacy B70_QSA_TOPK_SELECT_IMPL=triton ;;
lv-d13)    mk lv-d13    'D01 Triton select + D03 C2 scorer defaults' same B70_QSA_TOPK_SELECT_IMPL=triton ;;
lv-d13f)   mk lv-d13f   'lv-d13 + B70_QSA_TOPK_SKIP_FIT=1 (exact set, short contexts)' same \
                        B70_QSA_TOPK_SELECT_IMPL=triton B70_QSA_TOPK_SKIP_FIT=1 ;;
lv-d123)   mk lv-d123   'lv-d13 + B70_QSA_TOPK_PER_STEP=1 (APPROXIMATE past 2K tokens: fidelity gate first)' same \
                        B70_QSA_TOPK_SELECT_IMPL=triton B70_QSA_TOPK_PER_STEP=1 ;;
lv-d13nb)  mk lv-d13nb  'lv-d13 + B70_QSA_EAGER_BREAK=0 (QSA body captured; native + select inside the graph)' same \
                        B70_QSA_TOPK_SELECT_IMPL=triton B70_QSA_EAGER_BREAK=0 ;;
lv-d13prof) mk lv-d13prof 'lv-d13 + torch profiler (/start_profile, 40 iterations)' prof_cmd B70_QSA_TOPK_SELECT_IMPL=triton ;;
*) echo "unknown arm $1" >&2; exit 2 ;;
esac
}
mkdir -p $C/prof
if [ "${1:-all}" = all ]; then
  for s in lv-base lv-d01 lv-d01t lv-d13 lv-d13f lv-d123 lv-d13nb lv-d13prof; do run $s; done
else for s in "$@"; do run "$s"; done; fi
