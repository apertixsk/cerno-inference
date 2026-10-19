<div align="center">
  <img src="assets/cerno-light.jpg" alt="Cerno decision models" width="760"/>
  <h1>Cerno inference</h1>
  <p><b>Jev-style System One decision models in three sizes &middot; one forward pass &middot; calibrated probabilities &middot; FP8</b></p>
  <p>
    <a href="https://huggingface.co/apertixai/cerno-27b">Cerno 27B</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-9b">Cerno 9B</a> &middot;
    <a href="https://huggingface.co/apertixai/cerno-4b">Cerno 4B</a> &middot;
    <a href="https://huggingface.co/datasets/apertixai/cerno-decision-index">Decision Index runs</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Decision%20Index%200.3-27B%2065.19%20%C2%B7%209B%2058.05%20%C2%B7%204B%2056.58-1f6feb" alt="Decision Index 0.3 public index"/>
    <img src="https://img.shields.io/badge/weights-FP8%20W8A8-6f42c1" alt="FP8 weights"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-lightgrey" alt="Apache 2.0"/>
  </p>
</div>

Inference code for the **Cerno** family: Jev-style *System One* decision models in three sizes. A Cerno model takes a
**state** (any text or JSON) and a set of **typed questions** about it (multiple choice with explicit criteria, yes/no, scored
scales) and returns, in **one forward pass with no text generation**, a calibrated probability for every option of every
question. This repository holds the engine that produced the family's Decision Index runs and a two-minute check that
reproduces them.

## The family

| model | base | weights (FP8, vLLM) | temperature | Decision Index 0.3 public index | same-subset rank (145 entrants) | latency, one H200 (median / mean / p80) |
|---|---|---|---|---|---|---|
| **Cerno 27B** | Qwen3.8-27B | [`apertixai/cerno-27b`](https://huggingface.co/apertixai/cerno-27b) | 1.0 | **65.19** | **2nd** | 52 / 206 / 332 ms |
| **Cerno 9B** | Qwen3.5-9B | [`apertixai/cerno-9b`](https://huggingface.co/apertixai/cerno-9b) | 1.539 | **58.59** | 14th, first among entrants of declared size 10B or under | 30 / 91 / 141 ms |
| **Cerno 4B** | Qwen3.5-4B | [`apertixai/cerno-4b`](https://huggingface.co/apertixai/cerno-4b) | 1.534 | **57.60** | 25th, first among entrants of declared size 5B or under | 31 / 83 / 115 ms |

Public index = the complete 0.3 public suite (37 benchmarks, 140,178 requests) run with the maintainers' kit at commit
9eb2dbe through this code and the FP8 weights. The public suite is 20% of the board's headline Full score; the private 80%
is run by the maintainers on submitted models. Latency is one process, one request at a time, on the kit's 400-request
stratified sample. Each model's temperature ships in its `decider_config.json` and the engine applies it.

On-device builds (4-bit GGUF for llama.cpp, Q4_0 body with Q8_0 head, quantisation-aware trained): [`cerno-9b-GGUF`](https://huggingface.co/apertixai/cerno-9b-GGUF) (57.96 on the public index through llama.cpp), [`cerno-4b-GGUF`](https://huggingface.co/apertixai/cerno-4b-GGUF) (57.23).

v1.1 (2026-10-19): new 9B and 4B weights (+0.5 and +1.0 on the public index; the 27B is unchanged); the llama.cpp path
(`engine_llama.py`) now samples the read-out token greedily, so llama-server always returns the pre-sampling probabilities
(temperature-1 sampling occasionally produced a partial UTF-8 byte token and lost them). The vLLM engine is unchanged.

## Quick start

One GPU (24 GB for the 4B and 9B, 40 GB or more for the 27B), Python 3.12, tested on H200.

```sh
git clone https://github.com/apertixsk/cerno-inference && cd cerno-inference
python -m venv .venv && . .venv/bin/activate         # or a conda env with python=3.12
pip install -r requirements.txt                      # vLLM from PyPI, the Decision Index kit at 9eb2dbe, nothing else
export PYTHONPATH=$PWD:$PWD/third_party/decider
```

### Check it in two minutes

```sh
bash run_sample.sh apertixai/cerno-27b               # or apertixai/cerno-9b, apertixai/cerno-4b
```

The script draws 60 Decision Index requests from *your* staged suite with the kit's own sampler (`suite sample --seed 7`; no
benchmark text is shipped here), scores them with the vLLM engine (77 requests including linked chunks, 294 answer fields)
and compares the answers with the bundled reference from our submitted run of that model (`sample/reference-<tier>.jsonl`).
Expected: identical answers on 99% or more of the fields, probabilities within a few hundredths. A fresh clone on our H200s
reproduced the 4B run on 294 of 294 fields with a mean probability difference of 0.001.

### Run the Decision Index

```sh
python -m decision_index pipeline --edition 0.3 --engine evals.decision_index.engine_vllm:VllmEngine \
    --option model=apertixai/cerno-27b --out runs/cerno-27b          # same for cerno-9b and cerno-4b
```

`pipeline` expects the suite staged as in the kit's README (`suite rebuild` + `suite import`). Engine options: `model` (Hub id
or local directory), `max_model_len` (default 20480; a longer row is declared unsupported, nothing is truncated),
`gpu_memory_utilization` (default 0.9), `temperature` (default: the model's `decider_config.json`). The engine records what
the weights are (`quantization` in `environment.json`, read from the checkpoint's `config.json`).

## Examples

Two requests from our own evaluation set (never benchmark rows), answered by the three models through the engine. The
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

| question | Cerno 27B | Cerno 9B | Cerno 4B |
|---|---|---|---|
| `kind` | other (0.94) | other (0.91) | other (0.78) |
| `urgency` | level 1: can wait a few days (0.56) | level 1: can wait a few days (0.54) | level 0: no action needed (0.58) |
| `tone` | level 2: neutral (0.90) | level 2: neutral (0.89) | level 2: neutral (0.92) |
| `needs_human` | yes (0.73) | yes (0.57) | yes (0.57) |

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

| question | Cerno 27B | Cerno 9B | Cerno 4B |
|---|---|---|---|
| `destructive` | yes (0.99) | yes (0.99) | yes (0.97) |
| `matches_request` | yes (0.90) | yes (0.94) | yes (0.90) |
| `external_effect` | no (1.00) | no (0.98) | no (0.99) |
| `risk` | level 1: minor, easily reversible (0.76) | level 1: minor, easily reversible (0.69) | level 1: minor, easily reversible (0.70) |

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
two-option choice. No thinking, no second reading, no ensembling, no abstention.

## Layout

```
evals/decision_index/engine_vllm.py    the served engine (vLLM, FP8 weights): what made the submitted runs
evals/decision_index/engine.py         bf16 HF/PyTorch engine (needs a bf16 checkpoint; reference path)
evals/decision_index/engine_llama.py   GGUF through llama-server (device parity only; several times slower on this architecture)
evals/engines.py                       loads the inference wrapper
third_party/decider/                   the inference subset of the open decider package (Apache-2.0): prompt layout, label table, readout
sample/                                reference-{27b,9b,4b}.jsonl, our answers on the 60 sampled requests
run_sample.sh, compare_runs.py         the two-minute check
examples/                              two example requests, the published answers of the three models, a script that re-runs them
```

Apache-2.0. Not affiliated with TypeSafe AI or with the Decision Index maintainers.
