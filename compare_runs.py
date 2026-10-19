"""Compare a Decision Index run (e.g. a quantised GGUF or FP8 path on a sample) with a reference run of the same checkpoint:
answer agreement, probability differences, and the run's own single-request latency distribution.
  python evals/decision_index/compare_runs.py runs/latency/<name>/results.jsonl runs/dindex03_T/<ref>/results.jsonl [--label x]"""
import argparse, json, statistics as st


def load(path):
    out = {}; ms = []; status = {}
    for line in open(path):
        r = json.loads(line); status[r["status"]] = status.get(r["status"], 0) + 1
        if r["status"] == "ok":
            out[r["run_id"]] = r["response"]["answers"]
            if r.get("model_request_wall_ms") is not None: ms.append(r["model_request_wall_ms"])
    return out, ms, status


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("ref"); ap.add_argument("--label", default=None); a = ap.parse_args()
    new, ms, status = load(a.run); ref, _, _ = load(a.ref)
    n = same = 0; diffs = []; noul = []
    for rid, ans in new.items():
        o = ref.get(rid)
        if not o: continue
        for f, q in ans.items():
            oq = o.get(f) or {}
            if q.get("type") == "choice" and oq.get("type") == "choice":
                pn, po = q.get("probabilities") or {}, oq.get("probabilities") or {}
                if not pn or not po or set(pn) != set(po): continue
                n += 1; diffs.append(max(abs(pn[k] - po[k]) for k in pn)); same += (max(pn, key=pn.get) == max(po, key=po.get))
            elif q.get("type") == "noul" and oq.get("type") == "noul":
                n += 1; d = abs(float(q["noul"]) - float(oq["noul"])); diffs.append(d); same += ((float(q["noul"]) >= 0.5) == (float(oq["noul"]) >= 0.5)); noul.append(d)
    ms.sort(); q = lambda p: ms[max(0, int(p * len(ms)) - 1)] if ms else float("nan")
    lab = a.label or a.run.split("/")[-2]
    print(f"COMPARE {lab}: requests={len(new)} status={status} fields={n} same_answer={same} ({100*same/max(n,1):.1f}%) max|dp|={max(diffs) if diffs else float('nan'):.4f} mean|dp|={st.mean(diffs) if diffs else float('nan'):.4f} | latency ms: median={st.median(ms) if ms else float('nan'):.0f} mean={st.mean(ms) if ms else float('nan'):.0f} p80={q(0.8):.0f} p95={q(0.95):.0f} max={max(ms) if ms else float('nan'):.0f}")


if __name__ == "__main__": main()
