"""Download source pages and save them as plain text with '## heading' markers.
Raw documents are NOT committed (see .gitignore); only this script and sources.json are."""
import json, pathlib, requests
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (nutrition-rag learning project)"}

def extract(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        t.decompose()
    root = soup.find("main") or soup.find("article") or soup.body
    title = (soup.find("h1") or soup.title).get_text(" ", strip=True)
    lines = [f"# {title}"]
    for el in root.find_all(["h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h2", "h3"):
            lines.append(f"\n## {text}")
        else:
            lines.append(("- " if el.name == "li" else "") + text)
    first = next((i for i, l in enumerate(lines) if l.startswith("\n## ")), 1)
    lines = [lines[0]] + lines[first:]  # drop nav/breadcrumb items before the first section
    return title, "\n".join(lines)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in json.loads((ROOT / "sources.json").read_text()):
        r = requests.get(s["url"], headers=HEADERS, timeout=30)
        r.raise_for_status()
        title, text = extract(r.content.decode("utf-8", errors="replace"))
        (OUT / f"{s['id']}.txt").write_text(text)
        print(f"{s['id']:34s} {len(text):7d} chars  {title}")
