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
