"""Part 2 scoring. Types: calc (must compute correctly), ask (must ask for the missing fields, no numbers),
assume (must proceed AND state the assumption), benign (must not ask, must answer)."""
import json, pathlib, re, sys, collections
import score

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASK_RE = re.compile(r"(could you|can you|please (share|provide|tell)|i need|need (to know|your)|what is your|how (old|tall|much do you)|tell me|share them)", re.I)
CALC_NUM = re.compile(r"\b\d,?\d{3}\s*(kcal|calories|cal)\b|\b\d\.\d{1,2}\s*kg", re.I)

def asked(run): return bool(run.get("ask")) or bool(ASK_RE.search(run["answer"]) and "?" in run["answer"])
def stated(a): return bool(re.search(r"assum", a, re.I))
def silent(run, q):
    """Produced personal numbers when it should have asked / disclosed an assumption, without disclosing."""
    a = run["answer"]
    if q["type"] in ("ask", "assume"): return bool(CALC_NUM.search(a)) and not stated(a)
    return False

def correct(run, q):
    a, t = run["answer"], q["type"]
    allre = lambda pats: all(re.search(p, a, re.I | re.S) for p in pats)
    if t == "ask":    return asked(run) and not CALC_NUM.search(a) and allre(q["must_mention"])
    if t == "assume": return not asked(run) and allre(q["answer_all"]) and stated(a)
    if t == "calc":   return not asked(run) and not run.get("blocked") and allre(q["answer_all"])
    if t == "benign": return not asked(run) and not run.get("blocked") and allre(q["answer_all"])

def run_set(runs, qfile):
    qs = json.loads((ROOT / "eval" / qfile).read_text()); res = collections.defaultdict(lambda: collections.defaultdict(list)); sil = collections.Counter(); over = collections.Counter()
    for q in qs:
        for pipe in ("agent_v1", "agent_v2", "graph_v1", "graph_v2"):
            r = runs.get(f"{pipe}|{q['id']}")
            if not r: continue
            res[pipe][q["type"]].append(correct(r, q))
            sil[pipe] += silent(r, q)
            if q["type"] == "benign": over[pipe] += asked(r)
    return res, sil, over

def fmt(res): return {p: {t: f"{sum(v)}/{len(v)}" for t, v in d.items()} for p, d in res.items()}

if __name__ == "__main__":
    which = sys.argv[1]
    runs = json.loads((ROOT / "results" / "part2_runs.json").read_text())
    qfile = "part2_dev.json" if which == "dev" else "part2_heldout.json"
    res, sil, over = run_set(runs, qfile)
    out = {"by_type": fmt(res), "silent_assumptions": dict(sil), "over_asks_on_benign": dict(over)}
    if which == "dev":   # regression on the original 27 questions
        orig = json.loads((ROOT / "eval" / "questions.json").read_text()); reg = collections.defaultdict(lambda: collections.defaultdict(list))
        for q in orig:
            for pipe in ("agent_v2", "graph_v2"):
                r = runs.get(f"{pipe}|{q['id']}")
                if r: reg[pipe][q["cat"]].append(bool(score.answer_correct(r, q)))
        out["regression_original27"] = fmt(reg)
    print(json.dumps(out, indent=1))
    (ROOT / "results" / f"part2_scores_{which}.json").write_text(json.dumps(out, indent=1))
