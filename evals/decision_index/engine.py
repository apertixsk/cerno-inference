"""Cerno adapter for the Decision Index harness (third_party/decision-index).

    python -m decision_index run --engine evals.decision_index.engine:CernoEngine --option model=<bf16 checkpoint dir or Hub id> [--option revision=<commit>] --out <dir> [--rows <shard>]

One request = one state + its typed questions -> one packed decider prompt, one forward pass, letter-logit readout.
Questions are `choice` (criteria = {key: description}), `noul` (yes/no) and `score` (criteria = ordered level descriptions); choices with >255 options are Unsupported
(decider's wide rendering limit), as are fewer than 2. Response is the TypeSafe systemone shape the harness validates.
v1.2 checkpoints declare in decider_config.json block-bidirectional attention (applied through transformers' block mask; vLLM
cannot apply it, so they run here) and read twice: a request of at most `read_twice_tokens` tokens is rendered as the state, the
questions with their options, "Read the same input again before answering:" and the state again (the layout of KnowLine Gen5.5
and FlexDM; Xu et al. 2023). `--option read_twice_tokens=0` turns it off."""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (ROOT, os.path.join(ROOT, "third_party", "decider"), os.path.join(ROOT, "third_party", "decision-index")):
    if p not in sys.path: sys.path.insert(0, p)
from decision_index.engines.base import Engine, Unsupported


class CernoEngine(Engine):
    name = "cerno"
    read_twice_tokens = 0   # class default: subclasses that skip CernoEngine.__init__ (vLLM, llama.cpp; v1.0 / v1.1) never read twice
    latency = ("Device-synchronized in-process request wall time: state + all questions packed into one prompt, one forward "
               "pass, letter-logit readout; excludes model loading.")

    def __init__(self, model=None, temperature=None, max_state_tokens=32768, read_twice_tokens=None, revision=None, **options):
        super().__init__(**options)
        if not model: raise ValueError("CernoEngine needs --option model=<checkpoint dir or Hub id>")
        import torch
        from evals.engines import load_engine
        hub_id = None
        if revision:   # a Hub id pinned to one commit: weights, tokenizer and decider_config.json all come from that snapshot
            from huggingface_hub import snapshot_download
            hub_id, model = model, snapshot_download(model, revision=revision)
        self.torch = torch; self.model_path = model; self.max_state_tokens = int(max_state_tokens); self.split_tokens = int(options.pop("split_tokens", 16000)) if "split_tokens" in options else 16000
        self.d = load_engine(model, temperature=float(temperature) if temperature is not None else None)
        tag = os.path.basename(os.path.dirname(model.rstrip("/"))) if model.rstrip("/").endswith("/model") else os.path.basename(model.rstrip("/"))
        self.tag = hub_id.split("/")[-1] if hub_id else tag
        self.read_twice_tokens = int(read_twice_tokens) if read_twice_tokens is not None else int(getattr(self.d, "read_twice_tokens", 0))   # option, else decider_config.json, else off
        self.attention = getattr(self.d, "attention", "causal")
        self.temperature = float(getattr(self.d, "T", None) or getattr(self.d, "temperature", None) or 1.0)   # effective readout temperature (option, else decider_config.json next to the weights, else 1.0)
        self.provenance = {"kind": "local-weights", "temperature": self.temperature, "attention": self.attention, "read_twice_tokens": self.read_twice_tokens,
                           "technique": "full fine-tune of a Qwen3.5 / Qwen3.8 model, decider recipe (state-first prompt, letter-logit readout)"
                                        + ("; block-bidirectional attention in the full-attention layers (context block, one block per question)" if self.attention == "block" else ""),
                           "model_path": hub_id or model, "revision": revision, "weights_repo": "huggingface.co/apertixai (cerno-27b, cerno-9b, cerno-4b, cerno-9b-v1.2, cerno-4b-v1.2)",
                           "policy": "state and all questions sent as one packed prompt (prompts over the trained 16k-token context are answered in sub-packs); one forward pass per pack; choice = softmax over option letters (<=255 options), noul = yes/no pair; no abstention"
                                     + (f"; requests of at most {self.read_twice_tokens} tokens are read twice" if self.read_twice_tokens else "")}

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
        if self.read_twice_tokens and self._est_tokens(state, qs) <= self.read_twice_tokens:
            state = self._read_twice(state, qs); self.read_twice_n = getattr(self, "read_twice_n", 0) + 1
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

    @staticmethod
    def _read_twice(state, qs):
        """Read twice (v1.2): the state, the questions with their options, "Read the same input again before answering:", then
        the state again; the packed questions and their answer slots follow as usual, so every token of the second copy comes
        after the whole input. Layout of KnowLine Gen5.5 and FlexDM (MIT), after RE2 (Xu et al. 2023, arXiv 2309.06275)."""
        import json as _j
        st = state if isinstance(state, str) else _j.dumps(state, ensure_ascii=False)
        lines = []
        for q in qs.values():
            if q["type"] == "choice":
                c = q.get("criteria") or {}
                opts = "; ".join(f"{k}: {v}" if v and v != k else str(k) for k, v in (c.items() if isinstance(c, dict) else ((x, "") for x in c)))
                lines.append(f"- {q.get('instructions', '')} Options: {opts}")
            elif q["type"] == "score":
                lines.append(f"- {q.get('instructions', '')} Levels: " + "; ".join(str(x) for x in (q.get("criteria") or [])))
            else:
                lines.append(f"- {q.get('instructions', '')} (yes or no)")
        return f"{st}\n\nQuestions:\n" + "\n".join(lines) + f"\n\nRead the same input again before answering:\n\n{st}"

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

def attention_of(model, revision=None):
    """The attention a checkpoint declares in its decider_config.json (local dir or Hub id): "block" for v1.2, else "causal"."""
    import json as _j, os as _o
    try:
        if _o.path.isdir(model):
            p = _o.path.join(model, "decider_config.json")
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(model, "decider_config.json", revision=revision)
        return _j.load(open(p)).get("attention", "causal")
    except Exception:
        return "causal"


def engine_for(model):
    """The engine a checkpoint needs: CernoEngine (transformers) for block attention, the vLLM engine otherwise."""
    if attention_of(model) != "causal":
        return CernoEngine
    from evals.decision_index.engine_vllm import VllmEngine
    return VllmEngine


def refuse_block_attention(model, engine_name):
    """v1.2 checkpoints use block-bidirectional attention, which this engine cannot apply: stop rather than run them causal."""
    attention = attention_of(model)
    if attention != "causal":
        raise ValueError(f"{model} uses {attention} attention, which {engine_name} cannot apply; run it with "
                         "--engine evals.decision_index.engine:CernoEngine (transformers, bf16)")
