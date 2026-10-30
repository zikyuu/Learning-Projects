"""Stage 3: the same RAG pipeline rebuilt with LangChain + Chroma.

Mapping back to the raw version (the point of this stage):
  chunk.py split_sections + split_long  ->  MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
  embed.py SentenceTransformer + normalise ->  HuggingFaceEmbeddings(model_name, encode_kwargs={'normalize_embeddings': True})
  numpy array + `vecs @ q` + argsort    ->  Chroma collection (HNSW index, cosine space) + retriever.invoke()
  top_k(k)                              ->  as_retriever(search_kwargs={'k': k})
  build_prompt + chat()                 ->  ChatPromptTemplate | ChatOllama | StrOutputParser   (LCEL pipe)
"""
import pathlib, re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chunk import BOILERPLATE, ROOT
from embed import MODEL_NAME
from rag import SYSTEM
import safety

def load_docs(max_chars=800, overlap=100):
    md = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=True)
    rc = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    out = []
    for f in sorted((ROOT / "data" / "raw").glob("*.txt")):
        for d in md.split_text(f.read_text()):
            if BOILERPLATE.match(d.metadata.get("section", "")) or len(d.page_content) < 40:
                continue
            for piece in rc.split_documents([d]):
                piece.page_content = f"{piece.metadata.get('title','')} - {piece.metadata.get('section','')}\n{piece.page_content}"
                piece.metadata["source"] = f.stem
                out.append(piece)
    return out

_emb = None
def build_store(max_chars=800, overlap=100):
    global _emb
    _emb = _emb or HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    return Chroma.from_documents(load_docs(max_chars, overlap), _emb, collection_metadata={"hnsw:space": "cosine"})

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", "SOURCES:\n{context}\n\nQUESTION: {question}")])

class LCRag:
    def __init__(self, max_chars=800, overlap=100, k=4):
        self.store = build_store(max_chars, overlap)
        self.retriever = self.store.as_retriever(search_kwargs={"k": k})
        self.chain = PROMPT | ChatOllama(model="llama3.1:8b", temperature=0, num_ctx=4096) | StrOutputParser()

    def search(self, q):
        return [{"id": d.metadata["source"], "source": d.metadata["source"], "text": d.page_content} for d in self.retriever.invoke(q)]

    def answer(self, question, use_guardrail=True):
        if use_guardrail:
            blocked, reason, msg = safety.check(question)
            if blocked:
                return {"answer": msg, "blocked": reason, "hits": []}
        hits = self.search(question)
        ctx = "\n\n".join(f"[{i+1}] ({h['id']}) {h['text']}" for i, h in enumerate(hits))
        return {"answer": self.chain.invoke({"context": ctx, "question": question}).strip(), "blocked": None, "hits": hits}

if __name__ == "__main__":
    r = LCRag(); out = r.answer("What is a realistic weight-loss goal in the first 6 months?")
    print(out["answer"]); print([h["id"] for h in out["hits"]])
