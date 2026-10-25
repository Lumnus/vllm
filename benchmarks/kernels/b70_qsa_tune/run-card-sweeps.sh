#!/bin/bash
# B70-QSA-T: the single-card part of the Cell C GPU program, one job per card in parallel (engine stopped,
# no other /dev/dri holder). Each job writes <OUT>/<job>.jsonl and a log; read them with jq or the C artifact's
# recipes. usage: run-card-sweeps.sh <phase>   phases: s0 (inventory) s1 (coarse sweeps) s2 (arm-main) s3 (refine)
set -euo pipefail
PHASE=${1:?phase}
TREE=${TREE:-/work/src/vllm-v032-qsa-tune}
VENV=${VENV:-/work/venvs/t214-line032pr2}
ARMMAIN=${ARMMAIN:-/work/cells/qsa-triton-A/venvs/qsa-arm-main}
OUT=${OUT:-/work/cells/qsa-triton-C/bench/$(date -u +%Y%m%dT%H%M%SZ)-$PHASE}
CR=${CR:-/opt/compute-runtime-26.35/run}
mkdir -p "$OUT"
B=$TREE/benchmarks/kernels/bench_qwen4_qsa_xpu.py
job() {  # job <card> <name> <venv> <args...>
  local card=$1 name=$2 v=$3; shift 3
  ( export PATH=$v/bin:$PATH LD_LIBRARY_PATH=$v/lib PYTHONPATH=$TREE ZE_FLAT_DEVICE_HIERARCHY=FLAT \
           TRITON_CACHE_DIR=/work/cells/qsa-triton-C/triton-cache-$(basename $v)-card$card TMPDIR=/work/l32t/qsac
    unset LD_PRELOAD; mkdir -p "$TMPDIR"
    $CR $v/bin/python -P "$B" --card "$card" --out "$OUT/$name.jsonl" "$@" > "$OUT/$name.log" 2>&1
    echo "$name rc=$?" >> "$OUT/DONE" ) &
}
case "$PHASE" in
s0)  # default configs, every op, eager and graph replay; native vs torch top-k inside
  job 0 inv-decode-eager $VENV --no-prefill --budget
  job 1 inv-decode-graph $VENV --no-prefill --graph --budget
  job 2 inv-prefill $VENV --no-decode --budget
  job 3 inv-decode-mml32k $VENV --no-prefill --max-model-len 32768 --ctx 4096,16384,32768 --budget ;;
s1)  # coarse sweeps (sparse attention first: the prefill region is the largest suspect)
  job 0 sw-gqa-prefill $VENV --sweep sparse_gqa_prefill --ctx 32768,131072 --reps 3
  job 1 sw-prefill-score $VENV --sweep prefill_score --ctx 32768,131072 --reps 3
  job 2 sw-decode-score $VENV --sweep decode_score --rows 8,64 --ctx 4096,131072 --graph --reps 3
  job 3 sw-gqa-decode $VENV --sweep sparse_gqa_decode --rows 8,64 --ctx 32768,131072 --graph --reps 3 ;;
s2)  # Intel triton main as an arm (same ops; decode scorer without padding where it compiles)
  job 0 arm-inv-decode $ARMMAIN --no-prefill --graph --budget
  job 1 arm-inv-prefill $ARMMAIN --no-decode --budget
  job 2 arm-decode-score $ARMMAIN --sweep decode_score --rows 8 --ctx 4096,131072 --graph --reps 3 \
        --grid-json '{"B70_QSA_MIN_DOT_N":["4","16"],"B70_QSA_ALLOW_SMALL_DOT":["1"]}'
  job 3 arm-gqa-prefill $ARMMAIN --sweep sparse_gqa_prefill --ctx 32768 --reps 3 ;;
s3)  # refine around the s1 winners: write the grids to $OUT/../grid-*.json first
  job 0 rf-gqa-prefill $VENV --sweep sparse_gqa_prefill --grid-json @${GRID_GQA_PREFILL:?} --ctx 8192,32768,131072,262144
  job 1 rf-prefill-score $VENV --sweep prefill_score --grid-json @${GRID_PREFILL_SCORE:?} --ctx 8192,32768,131072,262144
  job 2 rf-decode-score $VENV --sweep decode_score --grid-json @${GRID_DECODE_SCORE:?} --rows 8,32,64 --graph
  job 3 rf-gqa-decode $VENV --sweep sparse_gqa_decode --grid-json @${GRID_GQA_DECODE:?} --rows 8,32,64 --graph ;;
*) echo "unknown phase" >&2; exit 2 ;;
esac
wait
echo "all jobs done: $OUT"; cat "$OUT/DONE"
