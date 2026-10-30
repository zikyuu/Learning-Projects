"""Stage 5: run every pipeline over every question and cache raw outputs; scoring lives in score.py."""
import json, pathlib, time, sys
from retrieve import Retriever
import rag, agent
from lc_rag import LCRag
from graph_rag import GraphRag

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "eval_runs.json"

def main():
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())
    runs = json.loads(OUT.read_text()) if OUT.exists() else {}
    raw = Retriever(); lc = LCRag(); gr = GraphRag(lc)
    pipelines = {
        "raw_rag":   lambda q: rag.answer(q, raw),
        "raw_agent": lambda q: agent.run_agent(q, raw),
        "langchain": lambda q: lc.answer(q),
        "langgraph": lambda q: gr.answer(q),
    }
    for item in qs:
        for name, fn in pipelines.items():
            key = f"{name}|{item['id']}"
            if key in runs: continue
            t = time.time()
            try: out = fn(item["q"])
            except Exception as e: out = {"answer": f"[ERROR] {type(e).__name__}: {e}", "blocked": None}
            out["seconds"] = round(time.time() - t, 1)
            out["hits"] = [{"id": h.get("id"), "source": h.get("source") or h.get("id", "").split("#")[0], "text": h["text"]} for h in out.get("hits", [])]
            runs[key] = out
            OUT.write_text(json.dumps(runs, indent=1))
            print(f"{key:20s} {out['seconds']:6.1f}s  {out['answer'][:90]!r}", flush=True)

if __name__ == "__main__":
    main()
