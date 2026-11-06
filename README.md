<div align="center">
  <img src="assets/cerno-light.jpg" alt="Cerno decision models" width="760"/>
  <h1>Cerno inference</h1>
  <p><b>Jev-style System One decision models in three sizes &middot; one forward pass &middot; calibrated probabilities &middot; FP8 and bf16</b></p>
  <p>
    <a href="https://huggingface.co/apertixai/cerno-27b-v1.1">Cerno 27B v1.1</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-9b-v1.2">Cerno 9B v1.2</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-4b-v1.2">Cerno 4B v1.2</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-27b">27B v1.0</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-9b">9B v1.1</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-4b">4B v1.1</a> &middot;
    <a href="https://huggingface.co/datasets/apertixai/cerno-decision-index">Decision Index runs</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Decision%20Index%200.3-27B%20v1.1%2069.73%20%C2%B7%209B%20v1.2%2066.83%20%C2%B7%204B%20v1.2%2065.05-1f6feb" alt="Decision Index 0.3 public index"/>
    <img src="https://img.shields.io/badge/weights-FP8%20(v1.0%2C%20v1.1)%20%C2%B7%20bf16%20(v1.2)-6f42c1" alt="FP8 and bf16 weights"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-lightgrey" alt="Apache 2.0"/>
  </p>
</div>

Inference code for the **Cerno** family: Jev-style *System One* decision models in three sizes. A Cerno model takes a
**state** (any text or JSON) and a set of **typed questions** about it (multiple choice with explicit criteria, yes/no, scored
scales) and returns, in **one forward pass with no text generation**, a calibrated probability for every option of every
question. This repository holds the engine that produced the family's Decision Index runs and a two-minute check that
reproduces them.

## The family

| model | base | weights | engine | temperature | Decision Index 0.3 public index | latency, one H200 (median / mean / p80) |
|---|---|---|---|---|---|---|
| **Cerno 27B v1.1** | Qwen3.8-27B | [`apertixai/cerno-27b-v1.1`](https://huggingface.co/apertixai/cerno-27b-v1.1) | vLLM, FP8 | 1.059 | **69.73** | 100 / 231 / 426 ms |
| **Cerno 9B v1.2** | Qwen3.5-9B | [`apertixai/cerno-9b-v1.2`](https://huggingface.co/apertixai/cerno-9b-v1.2) | transformers, bf16 | 1.256 | **66.83** | 62 / 264 / 345 ms |
| **Cerno 4B v1.2** | Qwen3.5-4B | [`apertixai/cerno-4b-v1.2`](https://huggingface.co/apertixai/cerno-4b-v1.2) | transformers, bf16 | 1.236 | **65.05** | 61 / 208 / 266 ms |
| Cerno 27B v1.0 | Qwen3.8-27B | [`apertixai/cerno-27b`](https://huggingface.co/apertixai/cerno-27b) | vLLM, FP8 | 1.0 | 65.19 | 52 / 206 / 332 ms |
| Cerno 9B v1.1 | Qwen3.5-9B-Base | [`apertixai/cerno-9b`](https://huggingface.co/apertixai/cerno-9b) | vLLM, FP8 | 1.539 | 58.59 | 30 / 91 / 141 ms |
| Cerno 4B v1.1 | Qwen3.5-4B-Base | [`apertixai/cerno-4b`](https://huggingface.co/apertixai/cerno-4b) | vLLM, FP8 | 1.534 | 57.60 | 31 / 83 / 115 ms |

Public index = the complete 0.3 public suite (37 benchmarks, 140,178 requests) run with the maintainers' kit at commit
9eb2dbe through this code and the weights listed. The board (Decision Index 0.4) ranks by its Full score: private tests of
the same skills (60%) and private decision tasks from new domains (40%), run by the maintainers on submitted models; the
public index is shown next to it but does not count. Latency is one process, one request at a time, on the kit's
400-request stratified sample. Each model's settings ship in its `decider_config.json` and the engine applies them.

On-device builds (4-bit GGUF for llama.cpp, Q4_0 body with Q8_0 head, quantisation-aware trained): [`cerno-9b-GGUF`](https://huggingface.co/apertixai/cerno-9b-GGUF) (57.96 on the public index through llama.cpp), [`cerno-4b-GGUF`](https://huggingface.co/apertixai/cerno-4b-GGUF) (57.23).

v1.1 (2026-10-19): new 9B and 4B weights (+0.5 and +1.0 on the public index; the 27B is unchanged); the llama.cpp path
(`engine_llama.py`) now samples the read-out token greedily, so llama-server always returns the pre-sampling probabilities
(temperature-1 sampling occasionally produced a partial UTF-8 byte token and lost them). The vLLM engine is unchanged.

v1.2 (2026-10-30): new 9B and 4B models (`cerno-9b-v1.2`, `cerno-4b-v1.2`), trained from the post-trained Qwen3.5
checkpoints with block-bidirectional attention in the full-attention layers and served with read twice (requests of up
to 3,000 tokens are read twice). vLLM cannot apply the block mask, so they run on the transformers engine
(`evals/decision_index/engine.py`, bf16) with `requirements-v1.2.txt`. Their `decider_config.json` sets `attention: block`,
`read_twice_tokens: 3000` and the temperature, and the engine applies all three. The vLLM and llama.cpp engines refuse a
block-attention checkpoint instead of running it causally. v1.0, v1.1 and their engines are unchanged.

27B v1.1 (2026-11-06): a new 27B (`cerno-27b-v1.1`), trained from the post-trained Qwen3.8-27B and served with read twice
(requests of up to 3,000 tokens are read twice); +4.54 on the public index over 27B v1.0. It keeps causal
attention, so it runs on the vLLM engine in FP8 with `requirements-27b-v1.1.txt` (vLLM 0.31.0, transformers 5.17.0). The vLLM
engine now reads `read_twice_tokens` from `decider_config.json` (or `--option read_twice_tokens=`); v1.0 and v1.1 declare none,
so they read once as before (their answers are unchanged).

## Quick start

One GPU (24 GB for the 4B and 9B, 40 GB or more for the 27B), Python 3.12, tested on H200.

```sh
git clone https://github.com/apertixsk/cerno-inference && cd cerno-inference
python -m venv .venv && . .venv/bin/activate         # or a conda env with python=3.12
pip install -r requirements.txt                      # vLLM from PyPI, the Decision Index kit at 9eb2dbe, nothing else
export PYTHONPATH=$PWD:$PWD/third_party/decider
```

For the v1.2 models install `requirements-v1.2.txt` instead: transformers 5.17 with the flash-linear-attention kernels,
no vLLM. For the 27B v1.1 install `requirements-27b-v1.1.txt` (vLLM 0.31.0 with transformers 5.17.0, as our runs used). `run_sample.sh`, `examples/run_examples.py` and `engine_for(model)` pick the right engine from the checkpoint.

### Check it in two minutes

```sh
bash run_sample.sh apertixai/cerno-27b-v1.1          # or apertixai/cerno-27b, apertixai/cerno-9b-v1.2, apertixai/cerno-4b-v1.2, ...
```

The script draws 60 Decision Index requests from *your* staged suite with the kit's own sampler (`suite sample --seed 7`; no
benchmark text is shipped here), scores them with the vLLM engine (77 requests including linked chunks, 294 answer fields)
and compares the answers with the bundled reference from our submitted run of that model (`sample/reference-<tier>.jsonl`).
Expected: identical answers on 99% or more of the fields, probabilities within a few hundredths. A fresh clone on our H200s
reproduced the 4B run on 294 of 294 fields with a mean probability difference of 0.001.

### Run the Decision Index

```sh
python -m decision_index pipeline --edition 0.3 --engine evals.decision_index.engine_vllm:VllmEngine \
    --option model=apertixai/cerno-27b-v1.1 --out runs/cerno-27b-v1.1   # same for cerno-27b, cerno-9b and cerno-4b
```

The v1.2 models, through the transformers engine:

```sh
python -m decision_index pipeline --edition 0.3 --engine evals.decision_index.engine:CernoEngine \
    --option model=apertixai/cerno-9b-v1.2 --option revision=<commit> --out runs/cerno-9b-v1.2   # same for cerno-4b-v1.2
```

`pipeline` expects the suite staged as in the kit's README (`suite rebuild` + `suite import`). Engine options: `model` (Hub id
or local directory), `max_model_len` (default 20480; a longer row is declared unsupported, nothing is truncated),
`gpu_memory_utilization` (default 0.9), `temperature` (default: the model's `decider_config.json`). The engine records what
the weights are (`quantization` in `environment.json`, read from the checkpoint's `config.json`).

## Examples

Two requests from our own evaluation set (never benchmark rows), answered by the models through their engines. The
requests, the published answers and a script that re-runs them are in `examples/`.

**Support triage: a one-line customer message**

```python
state = "hey do u have showers at the downtown location"
questions = {
    "kind": {"type": "choice", "instructions": "What kind of message is this?",
             "criteria": {"incident": "an outage or something broken in production", "billing": "invoices, charges, refunds", "account": "cancellations, plan changes, access", "feedback": "praise or suggestions", "spam": "unsolicited or scam", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
             "criteria": ["no action needed", "can wait a few days", "should be handled today", "needs attention right now"]},
    "tone": {"type": "score", "instructions": "What is the sender's tone?",
             "criteria": ["hostile", "negative", "neutral", "positive"]},
    "needs_human": {"type": "noul", "instructions": "Does this message need a human to respond rather than an automated reply?"}
}
```

| question | Cerno 27B v1.1 | Cerno 27B | Cerno 9B v1.1 | Cerno 4B v1.1 | Cerno 9B v1.2 | Cerno 4B v1.2 |
|---|---|---|---|---|---|---|
| `kind` | other (0.92) | other (0.94) | other (0.91) | other (0.78) | other (0.95) | other (0.90) |
| `urgency` | level 1: can wait a few days (0.59) | level 1: can wait a few days (0.56) | level 1: can wait a few days (0.54) | level 0: no action needed (0.58) | level 2: should be handled today (0.48) | level 0: no action needed (0.50) |
| `tone` | level 2: neutral (0.92) | level 2: neutral (0.90) | level 2: neutral (0.89) | level 2: neutral (0.92) | level 2: neutral (0.96) | level 2: neutral (0.97) |
| `needs_human` | no (0.59) | yes (0.73) | yes (0.57) | yes (0.57) | no (0.59) | no (0.60) |

**Agent guardrail: should this tool call run without asking?**

```python
state = "rm -rf /tmp/build-cache-7f3a2e   # user asked to clear the build cache before a rebuild"
questions = {
    "destructive": {"type": "noul", "instructions": "Would executing this tool call delete, overwrite or irreversibly change data?"},
    "matches_request": {"type": "noul", "instructions": "Is this tool call a reasonable way to do what the user asked?"},
    "external_effect": {"type": "noul", "instructions": "Does this tool call affect people or systems outside the local machine (sends messages, moves money, publishes)?"},
    "risk": {"type": "score", "instructions": "How risky is it to run this tool call without asking the user first?",
             "criteria": ["harmless, read-only", "minor, easily reversible", "significant or hard to reverse", "severe or irreversible"]}
}
```

| question | Cerno 27B v1.1 | Cerno 27B | Cerno 9B v1.1 | Cerno 4B v1.1 | Cerno 9B v1.2 | Cerno 4B v1.2 |
|---|---|---|---|---|---|---|
| `destructive` | yes (0.98) | yes (0.99) | yes (0.99) | yes (0.97) | yes (0.96) | yes (0.93) |
| `matches_request` | yes (0.94) | yes (0.90) | yes (0.94) | yes (0.90) | yes (0.96) | yes (0.90) |
| `external_effect` | no (0.99) | no (1.00) | no (0.98) | no (0.99) | no (0.99) | no (0.98) |
| `risk` | level 1: minor, easily reversible (0.65) | level 1: minor, easily reversible (0.76) | level 1: minor, easily reversible (0.69) | level 1: minor, easily reversible (0.70) | level 1: minor, easily reversible (0.71) | level 1: minor, easily reversible (0.40) |

Programmatic use:

```python
from evals.decision_index.engine_vllm import VllmEngine

engine = VllmEngine(model="apertixai/cerno-9b")        # loads the FP8 weights and the shipped temperature
response, _ = engine(state, questions)              # one forward pass per question row, state prefix shared
response["answers"]["destructive"]                     # {"type": "noul", "noul": 0.99}
```

Question types: `choice` (criteria = option descriptions, `null` for none), `noul` (yes/no), `score` (ordered level descriptions).

## Protocol

One row per question: the state and that question rendered in the state-first layout, the rows of a request batched into one
vLLM generate call with the state prefix shared through the prefix cache, one generated token, the top-k logprobs read over
the option label tokens, renormalised, the shipped temperature applied as p^(1/T) (argmax-invariant). Yes/no questions are a
two-option choice. No thinking, no ensembling, no abstention. 27B v1.1: a request of up to 3,000 tokens is read twice (the
state, the questions with their options, "Read the same input again before answering:", the state again) and that
composite is the shared state of every row; a doubled prompt the served context cannot hold is read once. v1.0 and v1.1
read once.

v1.2 (`engine.py`): the state and every question packed into one prompt in the state-first layout, one forward pass. In
the full-attention layers each token attends both ways within its block (the state, or one question with its options) and
causally to the blocks before it (transformers' block mask); the linear-attention layers stay causal. A request of up to
3,000 tokens is read twice: the state, the questions with their options, "Read the same input again before answering:",
then the state again. The option-letter logits at each answer slot, softmax at the shipped temperature. Prompts over the
trained 16k-token context are answered in sub-packs that each fit it. No thinking, no ensembling, no abstention.

## Layout

```
evals/decision_index/engine_vllm.py    the served engine (vLLM, FP8 weights): what made the submitted runs
evals/decision_index/engine.py         transformers engine, bf16: the v1.2 models (block attention, read twice); reference path for v1.0 / v1.1
evals/decision_index/engine_llama.py   GGUF through llama-server (device parity only; several times slower on this architecture)
evals/engines.py                       loads the inference wrapper
third_party/decider/                   the inference subset of the open decider package (Apache-2.0): prompt layout, label table, readout
sample/                                reference-{27b,9b,4b}.jsonl, reference-{9b,4b}-v1.2.jsonl and reference-27b-v1.1.jsonl, our answers on the 60 sampled requests
requirements-v1.2.txt                  the v1.2 models' environment (transformers, no vLLM)
requirements-27b-v1.1.txt              the 27B v1.1 environment (vLLM 0.31.0, transformers 5.17.0)
run_sample.sh, compare_runs.py         the two-minute check
examples/                              two example requests, the published answers of the models, a script that re-runs them
```

Apache-2.0. Not affiliated with TypeSafe AI or with the Decision Index maintainers.
