"""Part 2 eval. Usage: python evaluate2.py dev   |  python evaluate2.py heldout   (heldout is run ONCE, after the code is frozen)."""
import json, pathlib, sys, time
from retrieve import Retriever
import agent
from lc_rag import LCRag
from graph_rag import GraphRag
from graph_tools import GraphTools

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "part2_runs.json"

def load(name): return json.loads((ROOT / "eval" / name).read_text())

def main(which):
    runs = json.loads(OUT.read_text()) if OUT.exists() else {}
    raw = Retriever(); lc = LCRag(); g1 = GraphRag(lc); g2 = GraphTools(lc)
    P = {"agent_v1": lambda q: agent.run_agent(q, raw),
         "agent_v2": lambda q: agent.run_agent(q, raw, guard_inputs=True),
         "graph_v1": lambda q: g1.answer(q),
         "graph_v2": lambda q: g2.answer(q)}
    jobs = []
    if which == "dev":
        jobs += [(n, q) for q in load("part2_dev.json") for n in P]
        jobs += [(n, q) for q in load("questions.json") for n in ("agent_v2", "graph_v2")]   # regression on the original 27
    elif which == "heldout":
        jobs += [(n, q) for q in load("part2_heldout.json") for n in P]
    for name, item in jobs:
        key = f"{name}|{item['id']}"
        if key in runs: continue
        t = time.time()
        try: out = P[name](item["q"])
        except Exception as e: out = {"answer": f"[ERROR] {type(e).__name__}: {e}", "blocked": None}
        out["seconds"] = round(time.time() - t, 1)
        out["hits"] = [{"id": h.get("id"), "source": h.get("source") or h.get("id", "").split("#")[0], "text": h["text"]} for h in out.get("hits", [])]
        runs[key] = out; OUT.write_text(json.dumps(runs, indent=1))
        print(f"{key:18s} {out['seconds']:6.1f}s ask={out.get('ask')} {out['answer'][:80]!r}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
