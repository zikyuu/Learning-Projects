"""Part 2: LangGraph with a deterministic calculator route and an ask-before-assume gate.
safety -> plan -> [calc? check_inputs -> (ask | calc)] -> [facts? retrieve -> grade -> (rewrite|generate)] -> compose"""
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import safety, inputs, calc, tools
from lc_rag import LCRag, PROMPT
from graph_rag import llm, SAFETY_PROMPT, MAX_RETRIES

class S(TypedDict):
    question: str; text: str; query: str; slots: dict; plan: dict; hits: List[dict]; relevant: bool; retries: int
    blocked: Optional[str]; ask: bool; calc_text: str; facts: str; answer: str; path: List[str]

def build(rag):
    def safety_node(s):
        blocked, reason, msg = safety.check(s["text"])
        if not blocked and not s["plan"]["calc"]:   # calculator questions: rules only (no LLM classifier needed for pure math)
            v = llm.invoke(SAFETY_PROMPT.format(q=s["question"])).content.strip().upper()
            if v.startswith("UNSAFE"): blocked, reason, msg = True, "llm-classifier", safety.REDIRECT.format(why="This looks like it needs individual professional advice.")
        return {"blocked": reason if blocked else None, "answer": msg or "", "path": s["path"] + ["safety"]}

    def plan_node(s):
        slots = inputs.extract(s["text"]); p = inputs.plan(s["text"], slots)
        return {"slots": slots, "plan": p, "query": inputs.facts_question(s["question"]) if p["calc"] else s["question"], "path": s["path"] + [f"plan({'calc' if p['calc'] else 'facts'})"]}

    def check_inputs(s):
        miss = inputs.missing(s["slots"], s["plan"])
        if miss: return {"ask": True, "answer": inputs.ask_message(miss, s["slots"]), "path": s["path"] + ["check_inputs(ASK)"]}
        return {"path": s["path"] + ["check_inputs(ok)"]}

    def calc_node(s):
        parts, p = [], s["plan"]
        try:
            if p["tdee"] or p["deficit"]:
                text, _ = calc.compute(s["slots"], p, s["question"]); parts.append(text)
        except tools.ToolError as e:
            return {"blocked": "low-intake-tool", "answer": calc.refusal_for_low_intake(e), "path": s["path"] + ["calc(REFUSED)"]}
        if p["food"]:
            food = llm.invoke(f"Extract only the food name from this question, nothing else.\nQuestion: {s['question']}\nFood:").content.strip().strip('."\'')
            if food and food.lower() in s["question"].lower():
                try:
                    r = tools.lookup_nutrition({"food": food}); parts.append(f"USDA FoodData Central, per 100 g of '{r['matched_food']}': {r['kcal']} kcal, protein {r['protein_g']} g, fat {r['fat_g']} g, carbs {r['carbs_g']} g.")
                except tools.ToolError as e: parts.append(f"Could not look up '{food}': {e}")
            else: parts.append("Which food do you mean? I couldn't identify it in your question.")
        return {"calc_text": "\n".join(parts), "path": s["path"] + ["calc"]}

    def retrieve(s):
        hits = rag.search(s["query"])
        if s["plan"]["calc"] and s["retries"] == 0:   # facts clause alone loses topic context: also search the full question minus numbers
            seen = {h["text"] for h in hits}
            hits += [h for h in rag.search(re.sub(r"\d[\d,\.]*", "", s["question"])) if h["text"] not in seen][:3]
        return {"hits": hits, "path": s["path"] + ["retrieve"]}
    def grade(s):
        ctx = "\n---\n".join(h["text"][:500] for h in s["hits"])
        if s["plan"]["calc"]:   # the personal-numbers part is computed by code; only judge the guideline/fact part
            q = f"Full question (context only): {s['question']}\nPart to check against the excerpts: {s['query']}"
        else: q = f"Question: {s['question']}"
        v = llm.invoke(f"Do these excerpts contain information that answers the question? Answer only YES or NO.\n\n{q}\n\nExcerpts:\n{ctx}\n\nAnswer:").content.strip().upper()
        return {"relevant": v.startswith("YES"), "path": s["path"] + ["grade"]}
    def rewrite(s):
        q = llm.invoke(f"Rewrite this question as a short keyword search query for a nutrition document database. Output only the query.\n\nQuestion: {s['question']}").content.strip().strip('"')
        return {"query": q, "retries": s["retries"] + 1, "path": s["path"] + ["rewrite"]}
    def generate(s):
        if not s["relevant"]: return {"facts": "I don't have that in my sources.", "path": s["path"] + ["generate(abstain)"]}
        ctx = "\n\n".join(f"[{i+1}] ({h['id']}) {h['text']}" for i, h in enumerate(s["hits"]))
        q = s["query"] if s["calc_text"] else s["question"]
        return {"facts": (PROMPT | llm).invoke({"context": ctx, "question": q}).content.strip(), "path": s["path"] + ["generate"]}
    def compose(s):
        ans = s["calc_text"] + (("\n\n" + s["facts"]) if s["facts"] else "")
        return {"answer": ans.strip(), "path": s["path"] + ["compose"]}

    g = StateGraph(S)
    for n, f in [("safety", safety_node), ("plan", plan_node), ("check_inputs", check_inputs), ("calc", calc_node), ("retrieve", retrieve),
                 ("grade", grade), ("rewrite", rewrite), ("generate", generate), ("compose", compose)]: g.add_node(n, f)
    g.set_entry_point("plan")                    # plan first so the LLM safety classifier can be skipped for pure calculator questions
    g.add_edge("plan", "safety")
    g.add_conditional_edges("safety", lambda s: END if s["blocked"] else ("check_inputs" if s["plan"]["calc"] else "retrieve"))
    g.add_conditional_edges("check_inputs", lambda s: END if s["ask"] else "calc")
    g.add_conditional_edges("calc", lambda s: END if s["blocked"] else ("retrieve" if s["plan"]["facts"] and not s["plan"]["food"] else "compose"))
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", lambda s: "generate" if s["relevant"] or s["retries"] >= MAX_RETRIES else "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", lambda s: "compose" if s["calc_text"] else END)
    g.add_edge("compose", END)
    return g.compile()

class GraphTools:
    def __init__(self, rag=None):
        self.rag = rag or LCRag(); self.app = build(self.rag)
    def answer(self, question, history=None):
        text = "\n".join((history or []) + [question])
        s = self.app.invoke({"question": question, "text": text, "query": question, "slots": {}, "plan": {"calc": False}, "hits": [], "relevant": False,
                             "retries": 0, "blocked": None, "ask": False, "calc_text": "", "facts": "", "answer": "", "path": []})
        # non-calc, non-blocked questions end at generate: surface that as the answer
        ans = s["answer"] or s["facts"]
        return {"answer": ans, "blocked": s["blocked"], "ask": s["ask"], "hits": s["hits"], "path": s["path"]}
