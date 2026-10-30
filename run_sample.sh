#!/bin/bash
# Two-minute check: score the same 60 Decision Index requests our submitted runs used (drawn from YOUR staged suite with the
# kit's own sampler, so no benchmark text lives in this repository) and compare the answers with the bundled reference for
# that model. Usage: bash run_sample.sh <model id or local dir> [suite dir, default suite-0.3] [reference file]
set -euo pipefail
MODEL=${1:-apertixai/cerno-27b}; SUITE=${2:-suite-0.3}; REF=${3:-}
if [ -z "$REF" ]; then case "$MODEL" in *27[bB]*) REF=sample/reference-27b.jsonl;; *9[bB]-v1.2*) REF=sample/reference-9b-v1.2.jsonl;; *4[bB]-v1.2*) REF=sample/reference-4b-v1.2.jsonl;; *9[bB]*) REF=sample/reference-9b.jsonl;; *4[bB]*) REF=sample/reference-4b.jsonl;; *) echo "cannot tell the tier from '$MODEL': pass the reference file as the third argument"; exit 1;; esac; fi
export PYTHONPATH=$PWD:$PWD/third_party/decider
[ -d "$SUITE" ] || { echo "stage the 0.3 suite first (decision_index suite rebuild + suite import, see the kit README); expected at $SUITE"; exit 1; }
python -m decision_index suite sample --edition 0.3 --dir "$SUITE" --n 60 --seed 7 --out sample/sample60.jsonl.gz
rm -rf runs/sample60
ENGINE=$(python -c "import sys; from evals.decision_index.engine import attention_of; print('evals.decision_index.engine:CernoEngine' if attention_of(sys.argv[1]) != 'causal' else 'evals.decision_index.engine_vllm:VllmEngine')" "$MODEL")
python -m decision_index run --edition 0.3 --suite-dir "$SUITE" --no-verify --engine "$ENGINE" --option model="$MODEL" --rows sample/sample60.jsonl.gz --out runs/sample60 --compact
python compare_runs.py runs/sample60/results.jsonl "$REF" --label "this machine vs our submitted run ($MODEL)"
