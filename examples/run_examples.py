"""Answer the example requests with a Cerno model and print the answers next to the published ones.
  python examples/run_examples.py apertixai/cerno-9b        (PYTHONPATH=<repo>:<repo>/third_party/decider, one GPU)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "third_party", "decider")]
from evals.decision_index.engine_vllm import VllmEngine
model = sys.argv[1] if len(sys.argv) > 1 else "apertixai/cerno-27b"
requests = json.load(open(os.path.join(HERE, "requests.json"))); published = json.load(open(os.path.join(HERE, "answers.json"))).get(model, {})
engine = VllmEngine(model=model)
for r in requests:
    response, _ = engine(r["state"], r["questions"]); print(f"\n== {r['name']}: {json.dumps(r['state'])[:100]}")
    for qid, a in response["answers"].items():
        ref = published.get(r["name"], {}).get(qid, {})
        if a["type"] == "choice": mine, theirs = f"{a['choice']} ({a['probabilities'][a['choice']]:.2f})", f"{ref.get('choice')} ({(ref.get('probabilities') or {}).get(ref.get('choice'), float('nan')):.2f})"
        elif a["type"] == "noul": mine, theirs = f"p(yes)={a['noul']:.2f}", f"p(yes)={ref.get('noul', float('nan')):.2f}"
        else: mine, theirs = f"level {max(range(len(a['probabilities'])), key=lambda j: list(a['probabilities'].values())[j])} (expected {a['score']:.2f})", f"expected {ref.get('score', float('nan')):.2f}"
        print(f"   {qid:16s} this run: {mine:28s} published: {theirs}")
