"""Stage 4: LangGraph version.  safety -> retrieve -> grade -> (generate | rewrite -> retrieve, max 2 retries)."""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import safety
from lc_rag import LCRag, PROMPT

MAX_RETRIES = 2
llm = ChatOllama(model="llama3.1:8b", temperature=0, num_ctx=4096)

class State(TypedDict):
    question: str          # original user question (never overwritten)
    query: str             # current search query (rewritten on retry)
    hits: List[dict]
    relevant: bool
    retries: int
    blocked: Optional[str]
    answer: str
    path: List[str]        # node trace, for the write-up

SAFETY_PROMPT = """Classify this message for a general nutrition-EDUCATION tool. Answer exactly UNSAFE if it (a) asks for
a personalised diet/meal plan, (b) mentions a medical condition, medication, pregnancy, or breastfeeding, (c) shows signs of
disordered eating, (d) targets very low calories or extreme/crash dieting, or (e) is about a child/teen. Otherwise answer SAFE.
Message: {q}\nAnswer (SAFE or UNSAFE):"""

def build_graph(rag: LCRag):
    def safety_node(s):
        blocked, reason, msg = safety.check(s["question"])           # layer 1: deterministic rules
        if not blocked:                                               # layer 2: LLM classifier for subtle cases
            verdict = llm.invoke(SAFETY_PROMPT.format(q=s["question"])).content.strip().upper()
            if verdict.startswith("UNSAFE"):
                blocked, reason, msg = True, "llm-classifier", safety.REDIRECT.format(why="This looks like it needs individual professional advice.")
        return {"blocked": reason if blocked else None, "answer": msg or "", "path": s["path"] + ["safety"]}

    def retrieve_node(s):
        return {"hits": rag.search(s["query"]), "path": s["path"] + ["retrieve"]}

    def grade_node(s):
        ctx = "\n---\n".join(h["text"][:500] for h in s["hits"])
        v = llm.invoke(f"Do these excerpts contain information that answers the question? Answer only YES or NO.\n\nQuestion: {s['question']}\n\nExcerpts:\n{ctx}\n\nAnswer:").content.strip().upper()
        return {"relevant": v.startswith("YES"), "path": s["path"] + ["grade"]}

    def rewrite_node(s):
        q = llm.invoke(f"Rewrite this question as a short keyword search query for a nutrition document database. Output only the query.\n\nQuestion: {s['question']}").content.strip().strip('"')
        return {"query": q, "retries": s["retries"] + 1, "path": s["path"] + ["rewrite"]}

    def generate_node(s):
        if not s["relevant"]:
            return {"answer": "I don't have that in my sources.", "path": s["path"] + ["generate(abstain)"]}
        ctx = "\n\n".join(f"[{i+1}] ({h['id']}) {h['text']}" for i, h in enumerate(s["hits"]))
        out = (PROMPT | llm).invoke({"context": ctx, "question": s["question"]}).content.strip()
        return {"answer": out, "path": s["path"] + ["generate"]}

    g = StateGraph(State)
    for name, fn in [("safety", safety_node), ("retrieve", retrieve_node), ("grade", grade_node),
                     ("rewrite", rewrite_node), ("generate", generate_node)]:
        g.add_node(name, fn)
    g.set_entry_point("safety")
    g.add_conditional_edges("safety", lambda s: END if s["blocked"] else "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", lambda s: "generate" if s["relevant"] or s["retries"] >= MAX_RETRIES else "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()

class GraphRag:
    def __init__(self, rag=None):
        self.rag = rag or LCRag()
        self.app = build_graph(self.rag)
    def answer(self, question):
        s = self.app.invoke({"question": question, "query": question, "hits": [], "relevant": False,
                             "retries": 0, "blocked": None, "answer": "", "path": []})
        return {"answer": s["answer"], "blocked": s["blocked"], "hits": s["hits"], "path": s["path"]}

if __name__ == "__main__":
    gr = GraphRag()
    for q in ["What is a realistic weight-loss goal in the first 6 months?", "What is the capital of France?", "I'm pregnant, what should I eat?"]:
        o = gr.answer(q); print("Q:", q, "\n  path:", " -> ".join(o["path"]), "\n  A:", o["answer"][:200], "\n")
