"""Decision Index engine over llama.cpp (GGUF): the deployment path. The one generated token is sampled greedily (its probabilities
are read pre-sampling from n_probs); sampling it at temperature 1 made llama-server drop the probabilities whenever the sampled token
was a partial UTF-8 byte (about 0.5% of requests). (phone / laptop / llama-server), used here to measure what
4/6/8-bit quantisation costs on the public index and what it buys in latency.

Protocol = decider's inference protocol: every question is scored in its own row (state + that question), the state prefix
is shared through llama-server's prompt cache, one generated token with neutral sampling, top-n probabilities read over the
option label tokens and renormalised (the same readout the app uses), the shipped temperature applied as
p^(1/T). Rows are rendered by decider's own `build` and decoded to text, so the server sees the exact training layout.
  --engine evals.decision_index.engine_llama:LlamaGGUFEngine --option model=<hf dir (tokenizer + decider_config.json)>
           --option gguf=<file.gguf> --option base_url=http://127.0.0.1:<port>"""
import json, math, os, random, time, urllib.error, urllib.request
from decision_index.engines.base import Unsupported
from evals.decision_index.engine import CernoEngine


class LlamaGGUFEngine(CernoEngine):
    name = "cerno_gguf"
    latency = "In-process request wall time: one llama-server /completion call per question (shared state prefix through the server's prompt cache), one generated token, top-n letter probabilities; excludes model loading."

    def __init__(self, model=None, gguf=None, base_url="http://127.0.0.1:8080", temperature=None, max_state_tokens=16384, n_probs=260, **options):
        from decision_index.engines.base import Engine
        Engine.__init__(self, **options)
        if not model or not gguf: raise ValueError("LlamaGGUFEngine needs --option model=<hf dir> --option gguf=<file.gguf>")
        from transformers import AutoTokenizer
        self.model_path = model; self.gguf = gguf; self.base_url = base_url.rstrip("/"); self.max_state_tokens = int(max_state_tokens); self.n_probs = int(n_probs)
        self.tok = AutoTokenizer.from_pretrained(model)
        cfg = {}
        try: cfg = json.load(open(os.path.join(model, "decider_config.json")))
        except Exception: pass
        self.temperature = float(temperature) if temperature is not None else float(cfg.get("temperature", 1.0))
        self.neutralize_none = bool(cfg.get("neutralize_none", True))
        from decider.prompt import label_table
        self.labels = label_table(self.tok)[0]
        self.tag = os.path.basename(gguf).replace(".gguf", "")
        self.provenance = {"kind": "local-weights-gguf", "temperature": self.temperature, "gguf": gguf, "technique": "Cerno checkpoint quantised with llama.cpp (llama-quantize); llama-server readout",
                           "policy": "one row per question (state + question), prompt cache shared across a request's questions, one token, top-n label probabilities renormalised, temperature as p^(1/T)"}
        import types; self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), OutOfMemoryError=MemoryError); self.splits = 0   # the base class checks torch.cuda after each request

    # the HF engine's token estimate uses its tokenizer lazily; reuse ours
    def _tok(self): return self.tok

    def _post(self, payload):
        req = urllib.request.Request(self.base_url + "/completion", data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:300]
                if e.code == 400 and "context" in body.lower(): raise Unsupported(f"prompt exceeds the server context: {body}")
                if attempt == 7: raise
                time.sleep(2 + 3 * attempt)
            except Exception:
                if attempt == 7: raise
                time.sleep(2 + 3 * attempt)

    def _row_probs(self, prompt, nopts):
        out = self._post(dict(prompt=prompt, n_predict=1, n_probs=max(self.n_probs, nopts + 2), temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, typical_p=1.0, repeat_penalty=1.0, cache_prompt=True))
        cp = out.get("completion_probabilities") or []
        if not cp: raise RuntimeError(f"no completion_probabilities: {json.dumps(out)[:300]}")
        top = cp[0]; cands = top.get("top_logprobs") or top.get("top_probs") or top.get("probs") or []
        raw = {}
        for c in cands:
            t = c.get("token", c.get("tok_str")); raw[t] = float(c["prob"]) if "prob" in c else math.exp(float(c["logprob"]))
        p = [raw.get(self.labels[j], 0.0) for j in range(nopts)]
        if self.temperature != 1.0: p = [x ** (1.0 / self.temperature) if x > 0 else 0.0 for x in p]
        s = sum(p)
        return [x / s for x in p] if s > 0 else [1.0 / nopts] * nopts

    def _run(self, state, qs):
        from decider.systemone import render_state, render_question, plan_rows, assemble
        from decider.infer import neutralize_options
        from decider.prompt import build, MAX_OPTIONS
        from decider.data.core import Example, Q
        ctx = render_state(state); rqs = {k: render_question(v) for k, v in qs.items()}
        opts = (lambda r: neutralize_options(r["options"])[0]) if self.neutralize_none else (lambda r: list(r["options"]))
        flat, index = plan_rows(rqs, False)
        class _Keep:
            def shuffle(self, x): pass
            def sample(self, xs, k): return xs[:k]
        probs = []
        for r in flat:
            it = build(Example(ctx, [Q(r["question"], opts(r), 0)], "index"), self.tok, _Keep(), max_options=MAX_OPTIONS, max_ctx_tokens=self.max_state_tokens, layout="state_first")
            prompt = self.tok.decode(it["ids"]); nopts = int(it["nopts"][0])
            probs.append(self._row_probs(prompt, nopts))
        return {"model": self.tag, "answers": assemble(rqs, index, probs)}

    def synchronize(self): pass

    def runtime(self):
        try:
            h = json.loads(urllib.request.urlopen(self.base_url + "/props", timeout=10).read())
        except Exception: h = {}
        return {"device": "llama-server", "gguf": os.path.basename(self.gguf), "server": {k: h.get(k) for k in ("n_ctx", "model_path", "build_info") if k in h}}
