#!/usr/bin/env bash
# B70-QSA-T (run runs/2026-W43/2026-10-25-qsa-triton-xpu, artifacts/C-tuning-prep.md): engine arms of the sparse QSA
# dev def, ONE change each (pattern of the v032 W2P mkdefs-w2p.sh). Refuses to overwrite unless OVERWRITE=1.
# Generated defs are NOT started (the engine owner starts/stops); dry-run each with FLASHNEXT_DRYRUN=1 first.
#   usage: mkdefs-qsa-tune.sh [all|<suffix>...]      BASE=<def> to change the base (default: torch top-k def)
set -euo pipefail
E=${E:-/etc/crusher/engines}
B=${BASE:-qsa-dev-intel-autoround-v032-dflash7-s8-kv128-bt4096-rcram-expplug-topktorch}
C=${CELL:-/work/cells/qsa-triton-C}
TREE=${TREE:-/work/src/vllm-v032-qsa-tune}            # checkout of github Lumnus/vllm b70/v0.32.0-qsa-tune
ARMMAIN=${ARMMAIN:-/work/cells/qsa-triton-A/venvs/qsa-arm-main}
for f in $B.env $B.cmd $B.args; do [ -e "$E/$f" ] || { echo "missing base $E/$f" >&2; exit 2; }; done
put() { local f=$E/$1; if [ -e "$f" ] && [ "${OVERWRITE:-0}" != 1 ]; then echo "exists: $f (OVERWRITE=1)" >&2; exit 2; fi
        cat > "$f"; echo "wrote $f"; }
# mk <suffix> <one-line change> <cmd filter> <env filter>
mk() {
  local N=$B-$1 change=$2 cmdf=$3 envf=$4
  { echo "# $N - Cell C 2026-10-25: = $B with ONE change: $change"; "$envf" < $E/$B.env; } | put $N.env
  "$cmdf" < $E/$B.cmd | put $N.cmd
  { echo "# $N - .args identical to $B.args"; cat $E/$B.args; } | put $N.args
}
same() { cat; }
addenv() { local line=$1; cat; echo "$line"; }
appendcmd() { local flags=$1; sed "s#\$# $flags#"; }
tree()     { sed "s#^PYTHONPATH=.*#PYTHONPATH=$TREE#"; }
native()   { grep -v '^B70_QSA_TOPK='; echo 'B70_QSA_TOPK=native'; }
trace()    { tree; echo "B70_STEP_TRACE=$C/lib/libb70_step_trace.so"; echo 'B70_STEP_TRACE_FUNCS=qsa'
             echo 'B70_STEP_TRACE_DEPTH=2'; echo 'B70_STEP_TRACE_EVERY=64'; echo "B70_STEP_TRACE_DIR=$C/trace"
             echo "VLLM_CACHE_ROOT=$C/cache-trace"; echo "TRITON_CACHE_DIR=$C/triton-trace"; }
prof_cmd() { appendcmd "--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$C/prof --profiler-config.torch_profiler_with_stack=false --profiler-config.max_iterations=40"; }
mml32k()   { appendcmd "--max-model-len 32768"; }
eager()    { appendcmd "--enforce-eager"; }
tune()     { tree; echo "B70_QSA_TUNE=@$C/tune.json"; }
noprep()   { echo 'B70_QSA_FUSED_PREPARE=0'; cat; }
nobreak()  { tree; echo "B70_QSA_EAGER_BREAK=0"; }
logits256(){ cat; echo 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256'; }
c1off()    { cat; echo 'VLLM_XPU_C1_TARGET_GRAPH=0'; }
armmain()  { tree | sed "s#^VENV=.*#VENV=$ARMMAIN#"; echo 'B70_QSA_ALLOW_SMALL_DOT=1'; }
armmain_n4(){ armmain; echo 'B70_QSA_MIN_DOT_N=4'; }

run() {
case "$1" in
native)    mk native  'B70_QSA_TOPK=native (W1 measured speed on the torch top-k only)' same native ;;
tune0)     mk tune0   "PYTHONPATH=$TREE (B70-QSA-T tree, no overrides: must equal the base)" same tree ;;
trace)     mk trace   'B70 step trace on (lib rebuilt for 0.32), QSA function marks, own caches' same trace ;;
prof)      mk prof    'torch profiler router (/start_profile), XPU activity, 40 iterations' prof_cmd same ;;
mml32k)    mk mml32k  '--max-model-len 32768 (page-table width 8,192 cols instead of 65,728: test of H1)' mml32k same ;;
eager)     mk eager   '--enforce-eager (graph contribution)' eager same ;;
tune)      mk tune    "B70_QSA_TUNE=@$C/tune.json (sweep winners) + PYTHONPATH=$TREE" same tune ;;
noprep)    mk noprep  'B70_QSA_FUSED_PREPARE=0' same noprep ;;
logits256) mk logits256 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 (KV capacity / prefill chunking)' same logits256 ;;
nobreak)   mk nobreak 'B70_QSA_EAGER_BREAK=0 + tree (QSA body captured in the breakable FULL segments)' same nobreak ;;
c1off)     mk c1off   'VLLM_XPU_C1_TARGET_GRAPH=0 (C1 target graph off: is QSA eager in the verify?)' same c1off ;;
armmain)   mk armmain "VENV=$ARMMAIN (Intel triton main 3.9.0+gitf552c356) + tree" same armmain ;;
armmain-n4) mk armmain-n4 'armmain + B70_QSA_MIN_DOT_N=4 (no decode-scorer padding)' same armmain_n4 ;;
*) echo "unknown arm $1" >&2; exit 2 ;;
esac
}
if [ "${1:-all}" = all ]; then
  for s in native tune0 trace prof mml32k eager tune noprep nobreak logits256 c1off armmain armmain-n4; do run $s; done
else for s in "$@"; do run "$s"; done; fi
