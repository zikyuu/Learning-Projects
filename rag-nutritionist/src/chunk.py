"""Stage 1a: section-aware chunking with overlap."""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent

BOILERPLATE = re.compile(r"^(review date|update date|references|alternative names|reviewed by|editorial team)", re.I)

def split_sections(text):
    """Return (doc_title, [(heading, body), ...]) using the '## ' markers from fetch_sources."""
    title = text.splitlines()[0].lstrip("# ").strip()
    parts = re.split(r"\n## ", "\n" + text.split("\n", 1)[1])
    sections = []
    for p in parts:
        if not p.strip():
            continue
        head, _, body = p.partition("\n")
        if BOILERPLATE.match(head.strip()):
            continue  # found via retrieval inspection: 'Review Date' chunks ranked highly for generic queries
        sections.append((head.strip(), body.strip()))
    return title, sections

def split_long(body, max_chars, overlap):
    """Greedy sentence packing; each new window re-uses the last `overlap` chars of the previous one."""
    if len(body) <= max_chars:
        return [body]
    sentences = re.split(r"(?<=[.!?])\s+|\n", body)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + " " if overlap else ""
        cur += s + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def chunk_document(text, source, max_chars=800, overlap=100):
    title, sections = split_sections(text)
    out = []
    for head, body in sections:
        for piece in split_long(body, max_chars, overlap):
            if len(piece) < 40:
                continue  # skip crumbs
            # Prepend title + heading so a chunk is self-describing once separated from its page.
            out.append({"source": source, "title": title, "section": head,
                        "text": f"{title} - {head}\n{piece}"})
    for i, c in enumerate(out):
        c["id"] = f"{source}#{i}"
    return out

def chunk_corpus(max_chars=800, overlap=100):
    chunks = []
    for f in sorted((ROOT / "data" / "raw").glob("*.txt")):
        chunks += chunk_document(f.read_text(), f.stem, max_chars, overlap)
    return chunks

if __name__ == "__main__":
    for mc in (400, 800, 1600):
        cs = chunk_corpus(mc, mc // 8)
        print(f"max_chars={mc:5d}: {len(cs):4d} chunks, mean {sum(len(c['text']) for c in cs)/len(cs):.0f} chars")
