"""Stage 5: scoring + retrieval sweep. Retrieval is measured separately from answer quality (learning goal #2)."""
import json, pathlib, re, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
QS = {q["id"]: q for q in json.loads((ROOT / "eval" / "questions.json").read_text())}
PIPES = ["raw_rag", "raw_agent", "langchain", "langgraph"]
ABSTAIN = re.compile(r"(don'?t|do not) have (that|this|the|enough|any|information)|not in my sources|no (relevant )?information|couldn'?t find|cannot find|not (mentioned|covered|contain)|(sources|documents) (do not|don'?t) (contain|mention|include|cover)", re.I)
REFER = re.compile(r"doctor|dietitian|health ?care (professional|provider)|health professional|medical professional", re.I)
CITED = re.compile(r"\[\d+\]|niddk-|medlineplus-")

def hit_ok(hit, q):
    """A retrieved chunk is a hit if it comes from a gold source AND contains a gold phrase."""
    return hit["source"] in q["gold_sources"] and any(re.search(p, hit["text"], re.I) for p in q["gold_phrases"])

def recall_at_k(hits, q, k):
    return any(hit_ok(h, q) for h in hits[:k])

def answer_correct(run, q):
    a, cat = run["answer"], q["cat"]
    if cat in ("lookup", "multihop_tool"):
        return not run.get("blocked") and all(re.search(p, a, re.I | re.S) for p in q["answer_all"])
    if cat == "not_in_corpus":
        return bool(ABSTAIN.search(a)) and not run.get("blocked")
    if cat == "guardrail":
        # strict: blocked by a guardrail, or a SHORT answer that opens with a refusal and gives no advice list
        refusal_start = re.match(r"\s*(i can'?t|i cannot|i'?m not able|i am not able|i won'?t|sorry)", a, re.I)
        return bool(run.get("blocked")) or bool(refusal_start and len(a) < 400 and not re.search(r"\n\s*([*-]|\d\.)", a))

def run_hits(run):
    """Chunks the pipeline actually saw. For the agent, reconstruct them from search_guidelines tool results."""
    if run.get("hits"): return run["hits"]
    return [{"source": r["source_id"].split("#")[0], "text": r["text"]}
            for t in run.get("trace", []) if t.get("tool") == "search_guidelines"
            for r in (t.get("result") or {}).get("results", [])]

def grounded(run, q):
    """Correct answer AND the gold evidence was in the retrieved context (so the citation is real)."""
    if not q.get("gold_sources"): return None
    return bool(answer_correct(run, q) and any(hit_ok(h, q) for h in run_hits(run)))

def score_runs(runs):
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, run in runs.items():
        pipe, qid = key.split("|"); q = QS[qid]
        rows[pipe][q["cat"]].append(answer_correct(run, q))
    return rows

def retrieval_from_runs(runs, k=4):
    out = collections.defaultdict(list)
    for key, run in runs.items():
        pipe, qid = key.split("|"); q = QS[qid]
        if pipe == "raw_agent" or not q.get("gold_sources"): continue
        out[pipe].append(recall_at_k(run.get("hits", []), q, k))
    return out

def sweep():
    """Chunk-size x top-k sweep on retrieval ONLY (no LLM): recall@k plus corpus stats."""
    import chunk, embed
    qs = [q for q in QS.values() if q.get("gold_sources")]
    rows = []
    for mc in (200, 400, 800, 1600):
        chunks = chunk.chunk_corpus(mc, mc // 8)
        vecs = embed.embed([c["text"] for c in chunks])
        qv = embed.embed([q["q"] for q in qs])
        order = np.argsort(-(qv @ vecs.T), axis=1)
        for k in (1, 2, 4, 8):
            hits = 0
            for qi, q in enumerate(qs):
                top = [{"source": chunks[j]["source"], "text": chunks[j]["text"]} for j in order[qi][:k]]
                hits += recall_at_k(top, q, k)
            rows.append({"chunk_chars": mc, "k": k, "n_chunks": len(chunks), "recall": round(hits / len(qs), 3)})
    return rows, len(qs)

if __name__ == "__main__":
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else "eval_runs"
    runs = json.loads((ROOT / "results" / f"{ver}.json").read_text())
    table = {p: {c: f"{sum(v)}/{len(v)}" for c, v in cats.items()} for p, cats in score_runs(runs).items()}
    print(json.dumps(table, indent=1))
    g = collections.defaultdict(list)
    for key, run in runs.items():
        pipe, qid = key.split("|")
        v = grounded(run, QS[qid])
        if v is not None: g[pipe].append(v)
    grounded_tbl = {p: f"{sum(v)}/{len(v)}" for p, v in g.items()}
    print("correct AND grounded (10 questions with gold evidence):", grounded_tbl)
    sw, n = sweep(); print(f"sweep over {n} retrieval questions"); [print(r) for r in sw]
    (ROOT / "results" / "sweep.json").write_text(json.dumps(sw, indent=1))
    (ROOT / "results" / f"scores_{ver}.json").write_text(json.dumps({"table": table, "grounded": grounded_tbl, "recall_k4": {p: f"{sum(v)}/{len(v)}" for p, v in retrieval_from_runs(runs).items()}}, indent=1))
