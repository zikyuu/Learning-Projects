"""Turn real captured outputs into PNGs for the write-up (terminal-style transcripts + charts).
The text in each transcript image is copied verbatim from the saved run outputs, not retyped."""
import json, pathlib, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"; IMG.mkdir(parents=True, exist_ok=True)
BG, FG, DIM, ACC, WARN, OK = "#1e1e2e", "#cdd6f4", "#7f849c", "#89b4fa", "#f38ba8", "#a6e3a1"

def terminal(name, title, segments, width=100):
    """segments: list of (text, style) with style in {'fg','dim','acc','warn','ok'}; long lines are wrapped."""
    col = {"fg": FG, "dim": DIM, "acc": ACC, "warn": WARN, "ok": OK}
    lines = []
    for text, style in segments:
        for raw in str(text).split("\n"):
            for w in (textwrap.wrap(raw, width, subsequent_indent="    ") or [""]):
                lines.append((w, col[style]))
    h = 0.30 * len(lines) + 0.9
    fig = plt.figure(figsize=(width * 0.108 + 0.6, h), facecolor=BG)
    fig.text(0.015, 1 - 0.28 / h, "●", color="#f38ba8", fontsize=11, va="center")
    fig.text(0.04, 1 - 0.28 / h, "●", color="#f9e2af", fontsize=11, va="center")
    fig.text(0.065, 1 - 0.28 / h, "●", color="#a6e3a1", fontsize=11, va="center")
    fig.text(0.5, 1 - 0.28 / h, title, color=DIM, fontsize=9, ha="center", va="center", family="monospace")
    for i, (t, c) in enumerate(lines):
        fig.text(0.02, 1 - (0.75 + 0.30 * i) / h, t, color=c, fontsize=9, family="monospace", va="center")
    fig.savefig(IMG / f"{name}.png", dpi=160, facecolor=BG); plt.close(fig)
    print("wrote", IMG / f"{name}.png")

def bar_chart(table, path="accuracy_by_category"):
    cats = ["lookup", "multihop_tool", "not_in_corpus", "guardrail"]
    pipes = ["raw_rag", "raw_agent", "langchain", "langgraph"]
    colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7"]
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.2
    for i, p in enumerate(pipes):
        vals = []
        for c in cats:
            a, b = table[p][c].split("/"); vals.append(100 * int(a) / int(b))
        xs = [j + (i - 1.5) * w for j in range(len(cats))]
        bars = ax.bar(xs, vals, w, label=p, color=colors[i])
        for x, v, c in zip(xs, vals, cats):
            ax.text(x, v + 1.5, table[p][c], ha="center", fontsize=7)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(["Lookup (8)", "Multi-hop / tool (7)", "Not in corpus (6)", "Guardrail (6)"])
    ax.set_ylabel("% answered correctly"); ax.set_ylim(0, 115); ax.legend(ncol=4, loc="upper center", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("Answer correctness by category and pipeline (llama3.1:8b, n=27)", fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / f"{path}.png", dpi=160); plt.close(fig); print("wrote", path)

def sweep_chart(rows, path="retrieval_sweep"):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for mc in sorted({r["chunk_chars"] for r in rows}):
        rs = [r for r in rows if r["chunk_chars"] == mc]
        ax.plot([r["k"] for r in rs], [100 * r["recall"] for r in rs], marker="o", label=f"chunk ≤{mc} chars ({rs[0]['n_chunks']} chunks)")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
    ax.set_xlabel("top-k retrieved"); ax.set_ylabel("recall@k (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Retrieval recall@k vs chunk size (no LLM involved)", fontsize=10)
    fig.tight_layout(); fig.savefig(IMG / f"{path}.png", dpi=160); plt.close(fig); print("wrote", path)
