# Local changes to the vendored decider subset

1. `decider/prompt.py` — state-first layout, narrow (<= 10 options) first question: the context and the first question block
   are tokenised **as one string** (`JOINT_FIRST_PIECE = True`) instead of two separately encoded id lists, so the tokens at
   the state -> question boundary are produced exactly as a single-string on-device runtime produces them. Context truncation
   to `max_ctx_tokens` still happens at the token level (then decoded back to text). Later questions, wide label sets and the
   schema-first layout are unchanged. Measured with the Qwen3.5 tokenizer on contexts ending in different characters: joint and
   separate encodings were identical in every case, so this is a guarantee rather than a fix.

2. The package is reduced to its inference subset (prompt layout and label table, typed-question rendering and answer
   assembly, the inference wrapper and its engines, the two example dataclasses). Everything else upstream ships (data
   pipeline, trainer, games, probes, evaluation, serving) is not part of this repository.

3. Block-bidirectional attention (v1.2 models): `decider/prompt.py` returns a block id per token (0 = the context, k + 1 = question
   k); `decider/model.py` collates them and, for checkpoints whose `decider_config.json` says `attention: block`, builds the
   per-layer-type masks transformers' Qwen3.5 backbone accepts: full-attention layers causal plus bidirectional within a block
   (`create_causal_mask(block_sequence_ids=...)`), linear-attention layers the 2-D padding mask; `decider/infer.py` reads the
   attention (and `read_twice_tokens`) from `decider_config.json` and keeps the CUDA-graph engine for causal checkpoints only.
   Causal checkpoints (v1.0, v1.1) are unaffected: their block ids are built and ignored.
