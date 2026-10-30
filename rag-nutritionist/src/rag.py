"""Stage 1d: raw RAG = retrieve -> build prompt -> generate, with citations."""
from retrieve import Retriever
from llm import chat
import safety

SYSTEM = """You are a nutrition EDUCATION assistant (not medical advice). Answer ONLY using the numbered sources below.
Rules:
- Cite every claim with its source number like [1] or [2].
- If the sources do not contain the answer, reply exactly: "I don't have that in my sources."
- Do not invent numbers. Keep the answer under 120 words.
- No personalised diet plans."""

def build_prompt(question, hits):
    ctx = "\n\n".join(f"[{i+1}] ({h['id']}) {h['text']}" for i, h in enumerate(hits))
    return f"SOURCES:\n{ctx}\n\nQUESTION: {question}"

def answer(question, retriever, k=4, use_guardrail=True):
    if use_guardrail:
        blocked, reason, msg = safety.check(question)
        if blocked:
            return {"answer": msg, "blocked": reason, "hits": []}
    hits = retriever.search(question, k)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, hits)}]
    reply = chat(msgs)["content"].strip()
    return {"answer": reply, "blocked": None, "hits": hits}

if __name__ == "__main__":
    import sys
    r = Retriever()
    q = " ".join(sys.argv[1:]) or "What is a realistic weight-loss goal?"
    out = answer(q, r)
    print("Q:", q, "\nA:", out["answer"])
    for i, h in enumerate(out["hits"], 1):
        print(f"  [{i}] {h['id']} ({h['score']:.2f})")
