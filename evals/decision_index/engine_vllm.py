"""Decision Index engine over vLLM (in-process `LLM`), for the FP8 fast path: decider's inference protocol
(one row per question, the state prefix shared through vLLM's prefix cache, the rows of one request batched into one generate
call), one generated token with top-k logprobs read over the option label tokens, renormalised, shipped temperature as p^(1/T).
  PYTHONPATH=<repo>:<repo>/third_party/decider:<repo>/third_party/decision-index, the `vllm` env
  --engine evals.decision_index.engine_vllm:VllmEngine --option model=<hf dir> [--option quantization=fp8] [--option max_model_len=20480]"""
import json, math, os, time
from decision_index.engines.base import Unsupported
from evals.decision_index.engine import CernoEngine


def stored_quantization(model, requested=None):
    """What the weights are, read from the checkpoint's config.json (local dir or Hub id): the compressed-tensors FP8 export
    reports itself; a plain bf16 checkpoint reports bf16; a vLLM-side --option quantization=... is reported as such."""
    if requested: return f"{requested} (applied by vLLM at load)"
    try:
        p = os.path.join(model, "config.json")
        if not os.path.exists(p):
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(model, "config.json")
        q = json.load(open(p)).get("quantization_config") or {}
    except Exception: q = {}
    if not q: return "bf16 (as stored)"
    fmt = q.get("format") or ""; groups = q.get("config_groups") or {}
    g = next(iter(groups.values()), {}) if isinstance(groups, dict) else {}
    w = (g.get("weights") or {}); a = (g.get("input_activations") or {})
    bits = f"W{w.get('num_bits', '?')}A{a.get('num_bits', '?')}" if w else ""
    return f"{q.get('quant_method', 'quantized')} {fmt} {bits} (as stored)".replace("  ", " ").strip()


class VllmEngine(CernoEngine):
    name = "cerno_vllm"
    latency = "In-process request wall time: the request's question rows batched into one vLLM generate call (prefix-cached state), one token each, top-k label logprobs; excludes model loading."

    def __init__(self, model=None, quantization=None, temperature=None, max_state_tokens=16384, max_model_len=20480, gpu_memory_utilization=0.9, **options):
        from decision_index.engines.base import Engine
        Engine.__init__(self, **options)
        if not model: raise ValueError("VllmEngine needs --option model=<hf dir>")
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.model_path = model; self.quantization = quantization or None; self.max_state_tokens = int(max_state_tokens); self.stored_quantization = stored_quantization(model, self.quantization)
        cfg = {}
        try:
            p = os.path.join(model, "decider_config.json")
            if not os.path.exists(p):   # a Hub id: the shipped temperature lives next to the weights there too
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(model, "decider_config.json")
            cfg = json.load(open(p))
        except Exception: pass
        self.temperature = float(temperature) if temperature is not None else float(cfg.get("temperature", 1.0))
        self.neutralize_none = bool(cfg.get("neutralize_none", True))
        t = time.time()
        self.llm = LLM(model=model, quantization=self.quantization, dtype="bfloat16", max_model_len=int(max_model_len), max_logprobs=300, gpu_memory_utilization=float(gpu_memory_utilization), enable_prefix_caching=True, trust_remote_code=True)
        self.tok = AutoTokenizer.from_pretrained(model)
        from decider.prompt import label_table
        self.label_ids = label_table(self.tok)[1]
        self.sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=300); self.sp_prime = SamplingParams(max_tokens=1, temperature=0.0)
        self.tag = os.path.basename(os.path.dirname(model.rstrip("/"))) if model.rstrip("/").endswith("/model") else os.path.basename(model.rstrip("/"))
        self.provenance = {"kind": "local-weights-vllm", "temperature": self.temperature, "quantization": self.stored_quantization, "loaded_seconds": round(time.time() - t),
                           "policy": "one row per question, rows of a request batched in one generate call with prefix caching, one token, top-k label logprobs renormalised, temperature as p^(1/T)"}
        import types; self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), OutOfMemoryError=MemoryError); self.splits = 0   # the base class checks torch.cuda after each request

    def _tok(self): return self.tok

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
        items = [build(Example(ctx, [Q(r["question"], opts(r), 0)], "index"), self.tok, _Keep(), max_options=MAX_OPTIONS, max_ctx_tokens=self.max_state_tokens, layout="state_first") for r in flat]
        longest = max(len(it["ids"]) for it in items)
        if longest + 1 > self.llm.llm_engine.model_config.max_model_len: raise Unsupported(f"row of {longest} tokens exceeds the served context")
        if len(items) > 1:                                        # prime the prefix cache with the rows' common prefix (the state), so the
            ids0 = items[0]["ids"]; L = min(len(it["ids"]) for it in items); k = 0   # batch below re-prefills it only once instead of once per row
            while k < L and all(it["ids"][k] == ids0[k] for it in items): k += 1
            if k >= 64: self.llm.generate([{"prompt_token_ids": ids0[:k]}], self.sp_prime, use_tqdm=False)
        outs = self.llm.generate([{"prompt_token_ids": it["ids"]} for it in items], self.sp, use_tqdm=False)
        probs = []
        for it, o in zip(items, outs):
            lp = o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}
            nopts = int(it["nopts"][0]); p = [math.exp(lp[tid].logprob) if tid in lp else 0.0 for tid in self.label_ids[:nopts]]
            if self.temperature != 1.0: p = [x ** (1.0 / self.temperature) if x > 0 else 0.0 for x in p]
            s = sum(p); probs.append([x / s for x in p] if s > 0 else [1.0 / nopts] * nopts)
        return {"model": self.tag, "answers": assemble(rqs, index, probs)}

    def synchronize(self): pass

    def runtime(self):
        import torch, vllm
        return {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "vllm": vllm.__version__, "quantization": self.stored_quantization}
