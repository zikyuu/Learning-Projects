"""Build every image in docs/img from saved results (results/*.json) and a live retrieval call."""
import json, pathlib
import render
from retrieve import Retriever

ROOT = pathlib.Path(__file__).resolve().parent.parent
v2 = json.loads((ROOT / "results" / "eval_runs.json").read_text())
v1 = json.loads((ROOT / "results" / "eval_runs_v1.json").read_text())
QS = {q["id"]: q["q"] for q in json.loads((ROOT / "eval" / "questions.json").read_text())}
P = lambda t: ("$ " + t, "acc")

# 1. charts
scores = json.loads((ROOT / "results" / "scores_eval_runs.json").read_text())
render.bar_chart(scores["table"])
render.sweep_chart(json.loads((ROOT / "results" / "sweep.json").read_text()))

# 2. raw RAG: retrieval + cited answer (live retrieval call, saved answer)
r = Retriever(); q = QS["L1"]; hits = r.search(q, 4); run = v2["raw_rag|L1"]
seg = [P("python src/retrieve.py   # cosine top-k over a plain numpy array"), (f"Q: {q}", "fg")]
seg += [(f"  {h['score']:.3f}  {h['id']:34s} [{h['section']}]", "dim") for h in hits]
seg += [("", "fg"), P("python src/rag.py  # retrieve -> prompt -> llama3.1:8b"), ("A: " + run["answer"], "ok")]
render.terminal("01_raw_rag_citation", "Stage 1 - raw RAG: retrieval scores + cited answer", seg)

# 3. agent success trace (M1)
a = v2["raw_agent|M1"]; seg = [P("python src/agent.py"), (f"Q: {QS['M1']}", "fg")]
for t in a["trace"]:
    if "nudge" in t: seg.append((f"  [{t['iter']}] nudge: model announced '{t['nudge']}' without calling it", "warn"))
    else: seg += [(f"  [{t['iter']}] {t['tool']}({json.dumps(t['args'])})", "acc"), (f"       -> {json.dumps(t['result'])[:200]}", "dim")]
seg += [("", "fg"), ("A: " + a["answer"], "ok"), (f"({a['iterations']} loop iterations; arithmetic done by code, not the LLM)", "dim")]
render.terminal("02_agent_tool_loop", "Stage 2 - hand-written agent loop", seg)

# 4. agent failure v0 (saved verbatim)
txt = (ROOT / "results" / "failure_agent_v0.txt").read_text().splitlines()
render.terminal("03_agent_failure_v0", "Stage 2 - failure case kept for the write-up", [(l, "warn" if "<-" in l else "fg") for l in txt])

# 5. guardrails: v1 vs v2
seg = [(f"Q (G6, no refusal keywords): {QS['G6']}", "fg"), ("", "fg")]
for name in ["raw_rag", "langchain", "langgraph"]:
    b = v1[name + "|G6"].get("blocked")
    seg += [(f"v1 {name:10s} blocked={b}", "ok" if b else "warn"), (f"      -> {v1[name+'|G6']['answer'][:150]}", "dim")]
seg += [("", "fg"), ("v2 (after adding a rule) every pipeline: blocked='unsafe restriction symptoms'", "ok"),
        ("      -> " + v2["raw_rag|G6"]["answer"][:170], "dim"), ("", "fg"),
        (f"Q (G1): {QS['G1']}", "fg"), (f"v2 raw_agent blocked={v2['raw_agent|G1']['blocked']}", "ok"), ("      -> " + v2["raw_agent|G1"]["answer"][:170], "dim")]
render.terminal("04_guardrails", "Guardrails: a rule miss caught by the LLM-classifier layer, then fixed", seg)

# 6. LangGraph path
seg = [P("python src/graph_rag.py   # safety -> retrieve -> grade -> (generate | rewrite -> retrieve) x max 2")]
for qid in ["L1", "N1", "G2"]:
    g = v2[f"langgraph|{qid}"]; seg += [(f"Q: {QS[qid]}", "fg"), ("  path: " + " -> ".join(g["path"]), "acc"), ("  A: " + g["answer"][:150], "ok"), ("", "fg")]
render.terminal("05_langgraph_paths", "Stage 4 - LangGraph: explicit state, conditional edges, retry cap", seg)

# 7. ungrounded-but-cited finding (L8)
g = v2["raw_rag|L8"]; seg = [(f"Q: {QS['L8']}", "fg"), ("A: " + g["answer"], "warn"), ("", "fg"), ("What the retriever actually returned (the BMI table was lost in HTML extraction):", "fg")]
for i, h in enumerate(g["hits"], 1):
    note = "30.7% (a prevalence stat, unrelated)" if "30.7" in h["text"] else "no"
    seg.append((f"  [{i}] {h['id']:28s} contains the BMI cut-off 30: {note}", "dim"))
seg += [("", "fg"), ("-> right answer, but NOT grounded: it came from model memory with a citation that does not support it.", "warn")]
render.terminal("06_ungrounded_citation", "Finding: a correct answer with a citation that does not support it", seg)

# 8. agent leaking outside knowledge
seg = []
for qid in ["N5", "N6", "M7"]:
    a = v2[f"raw_agent|{qid}"]; seg += [(f"Q: {QS[qid]}", "fg"), ("A: " + a["answer"][:260].replace("\n", " "), "warn"), ("", "fg")]
seg.append(("-> answers from model memory instead of abstaining (N5, N6), and cites 'WHO', which is not in the corpus (M7).", "warn"))
render.terminal("07_agent_knowledge_leak", "Finding: the agent mixes in knowledge that is not in the corpus", seg)
