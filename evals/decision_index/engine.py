"""Cerno adapter for the Decision Index harness (third_party/decision-index).

    python -m decision_index run --engine evals.decision_index.engine:CernoEngine --option model=<bf16 checkpoint dir> --out <dir> [--rows <shard>]

One request = one state + its typed questions -> one packed decider prompt, one forward pass, letter-logit readout.
Questions are `choice` (criteria = {key: description}), `noul` (yes/no) and `score` (criteria = ordered level descriptions); choices with >255 options are Unsupported
(decider's wide rendering limit), as are fewer than 2. Response is the TypeSafe systemone shape the harness validates."""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (ROOT, os.path.join(ROOT, "third_party", "decider"), os.path.join(ROOT, "third_party", "decision-index")):
    if p not in sys.path: sys.path.insert(0, p)
from decision_index.engines.base import Engine, Unsupported


class CernoEngine(Engine):
    name = "cerno"
    latency = ("Device-synchronized in-process request wall time: state + all questions packed into one prompt, one forward "
               "pass, letter-logit readout; excludes model loading.")

    def __init__(self, model=None, temperature=None, max_state_tokens=32768, **options):
        super().__init__(**options)
        if not model: raise ValueError("CernoEngine needs --option model=<checkpoint dir>")
        import torch
        from evals.engines import load_engine
        self.torch = torch; self.model_path = model; self.max_state_tokens = int(max_state_tokens); self.split_tokens = int(options.pop("split_tokens", 16000)) if "split_tokens" in options else 16000
        self.d = load_engine(model, temperature=float(temperature) if temperature is not None else None)
        tag = os.path.basename(os.path.dirname(model.rstrip("/"))) if model.rstrip("/").endswith("/model") else os.path.basename(model.rstrip("/"))
        self.tag = tag
        self.temperature = float(getattr(self.d, "T", None) or getattr(self.d, "temperature", None) or 1.0)   # effective readout temperature (option, else decider_config.json next to the weights, else 1.0)
        self.provenance = {"kind": "local-weights", "temperature": self.temperature, "technique": "full fine-tune of a Qwen3.5 / Qwen3.8 base, decider recipe (state-first prompt, letter-logit readout)",
                           "model_path": model, "weights_repo": "huggingface.co/apertixai (cerno-27b, cerno-9b, cerno-4b)",
                           "policy": "state and all questions sent as one packed prompt (prompts over the trained 16k-token context are answered in sub-packs); one forward pass per pack; choice = softmax over option letters (<=255 options), noul = yes/no pair; no abstention"}

    def __call__(self, state, questions):
        qs = {}
        for k, q in questions.items():
            t = q.get("type")
            if t == "choice":
                n = len(q.get("criteria") or {})
                if n < 2: raise Unsupported("a choice needs at least two options")
                if n > 255: raise Unsupported(f"choice with {n} options per choice exceeds the 255 the readout supports")
            elif t == "score":
                if len(q.get("criteria") or []) < 2: raise Unsupported("a score needs at least two levels")
            elif t != "noul":
                raise Unsupported("Unsupported question type " + str(t))
            qs[k] = {kk: q[kk] for kk in ("type", "instructions", "criteria") if kk in q}
        out = self._run(state, qs)
        if self.torch.cuda.is_available() and self._est_tokens(state, qs) > 6000:
            self.torch.cuda.empty_cache()   # long packs leave tens of GB in the allocator cache on a 27B; release it so the next one does not OOM
        answers = {}; self.degenerate = getattr(self, "degenerate", 0)
        for k, a in out["answers"].items():
            if a["type"] == "choice":
                p = {kk: float(v) for kk, v in a["probabilities"].items()}
                keys = list(qs[k]["criteria"]); s_ = sum(v for v in p.values() if v == v)
                if any(kk not in p or not (p[kk] == p[kk]) for kk in keys) or not (0.99 <= s_ <= 1.01):
                    # 1 in 150k requests: a degenerate slot (non-finite / unnormalised); fall back to uniform rather than lose the request
                    self.degenerate += 1; p = {kk: 1.0 / len(keys) for kk in keys}
                answers[k] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
            elif a["type"] == "score":
                answers[k] = {"type": "score", "score": float(a["score"]), "probabilities": a.get("probabilities")}
            else:
                v = float(a["noul"]); answers[k] = {"type": "noul", "noul": v if v == v else 0.5}
        return {"model": self.tag, "answers": answers}, out

    # ---- prompts longer than the trained context (16,384 tokens; 104 of 149,816 Index requests, all 32-question
    # ToolRet/BRIGHT/ACOS packs) are answered in sub-packs that each fit it. The 27B OOMs or faults in the kernels from ~20k
    # tokens where the 4B/9B happened to survive; the sub-pack keeps every prompt inside the range the models were trained
    # on. A sub-pack sees the full state and a subset of the questions; an out-of-memory on the way splits further, and a
    # single question that still does not fit is reported Unsupported (counted as unanswered) instead of halting the shard.
    def _tok(self):
        if getattr(self, "_tokenizer", None) is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def _est_tokens(self, state, qs):
        import json as _j
        txt = _j.dumps(state, ensure_ascii=False) + _j.dumps(qs, ensure_ascii=False)
        if len(txt) < 24000: return len(txt) / 3.5          # cannot exceed ~7k tokens: skip the tokenizer
        return len(self._tok()(txt, add_special_tokens=False)["input_ids"]) * 1.08   # rendered prompt adds ~8% (labels, letters, schema)

    def _run(self, state, qs):
        split_tokens = int(getattr(self, "split_tokens", 16000)); self.splits = getattr(self, "splits", 0)
        est = self._est_tokens(state, qs)
        if est > split_tokens and len(qs) > 1:
            keys = list(qs); n_chunks = min(len(keys), int(est // split_tokens) + 1); size = -(-len(keys) // n_chunks)
            parts = [keys[i:i + size] for i in range(0, len(keys), size)]; self.splits += 1
            merged = {}
            for part in parts: merged.update(self._run(state, {k: qs[k] for k in part})["answers"])
            return {"answers": merged}
        try:
            return self.d.system_one(state, qs, max_state_tokens=self.max_state_tokens)
        except self.torch.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(qs) == 1: raise Unsupported(f"out of memory on a single-question prompt of ~{int(est)} tokens")
            keys = list(qs); half = len(keys) // 2; self.splits += 1
            a = self._run(state, {k: qs[k] for k in keys[:half]}); b = self._run(state, {k: qs[k] for k in keys[half:]})
            return {"answers": {**a["answers"], **b["answers"]}}

    def synchronize(self):
        if self.torch.cuda.is_available(): self.torch.cuda.synchronize()

    def runtime(self):
        t = self.torch
        return {"device": t.cuda.get_device_name(0) if t.cuda.is_available() else "cpu", "torch": t.__version__, "dtype": "bf16"}
