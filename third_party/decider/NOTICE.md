# Vendored: Mapika/decider (inference subset)

Source: https://github.com/Mapika/decider  (commit in `VENDORED_FROM_COMMIT.txt`)
License: Apache-2.0 (declared in the upstream `pyproject.toml`).
Weights of the upstream release: https://huggingface.co/Mapika/decider-2b

What Cerno's inference code uses from it: the prompt layout and label table (`decider/prompt.py`), the typed-question
rendering and answer assembly (`decider/systemone.py`), the inference wrapper and its engines (`decider/infer.py`,
`decider/engine.py`, `decider/schema_engine.py`, `decider/model.py`, `decider/fp8.py`) and the two example dataclasses
(`decider/data/core.py`). The rest of the upstream package (data pipeline, trainer, games, probes, evaluation, serving)
is not part of this repository. Local modifications are listed in `LOCAL_CHANGES.md` next to this file.
