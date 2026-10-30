"""Part 2 figures from saved runs (results/part2_runs.json) + one live two-turn conversation (saved to results/part2_two_turn.json)."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import render
from graph_tools import GraphTools

ROOT = pathlib.Path(__file__).resolve().parent.parent
runs = json.loads((ROOT / "results" / "part2_runs.json").read_text())
Q = {q["id"]: q["q"] for f in ("part2_dev.json", "part2_heldout.json") for q in json.loads((ROOT / "eval" / f).read_text())}
held = json.loads((ROOT / "results" / "part2_scores_heldout.json").read_text())
dev = json.loads((ROOT / "results" / "part2_scores_dev.json").read_text())

# two-turn conversation (live, saved)
g = GraphTools(); t1 = g.answer(Q["D1"]); t2 = g.answer("I'm a 28 year old male, moderately active.", history=[Q["D1"]])
(ROOT / "results" / "part2_two_turn.json").write_text(json.dumps({"turn1": t1, "turn2": t2}, indent=1))
render.terminal("10_part2_two_turn", "Part 2 - LangGraph asks, then answers on the follow-up (history carried in state)", [
    ("You: " + Q["D1"], "fg"), ("  path: " + " -> ".join(t1["path"]), "acc"), ("Bot: " + t1["answer"], "ok"), ("", "fg"),
    ("You: I'm a 28 year old male, moderately active.", "fg"), ("  path: " + " -> ".join(t2["path"]), "acc"), ("Bot: " + t2["answer"], "ok")])

# invented inputs (baseline agent) vs ask (graph)
a, b = runs["agent_v1|H5"], runs["graph_v2|H5"]
render.terminal("08_part2_invented_inputs", "Part 2 - held-out: same question, baseline agent vs. ask-first graph", [
    ("Q: " + Q["H5"], "fg"), ("", "fg"), ("baseline agent (no age/weight/height were given):", "dim"),
    ("  tools: " + str([t.get("tool") for t in a["trace"]]), "dim"), ("  " + a["answer"], "warn"), ("", "fg"),
    ("graph with check_inputs gate:", "dim"), ("  path: " + " -> ".join(b["path"]), "acc"), ("  " + b["answer"], "ok")])

# assumption disclosed with range (dev, tuned on)
c = runs["graph_v2|D3"]
render.terminal("09_part2_assumption_range", "Part 2 - missing activity level: assume, disclose, show the range (dev question)", [
    ("Q: " + Q["D3"], "fg"), ("  path: " + " -> ".join(c["path"]), "acc"), ("", "fg"), (c["answer"], "ok")])

# held-out chart
pipes = ["agent_v1", "agent_v2", "graph_v1", "graph_v2"]; labels = ["raw agent\n(baseline)", "raw agent\n+ guard", "LangGraph\n(baseline)", "LangGraph\n+ calc route + ask"]
types = [("calc", "Calculation", 5), ("ask", "Missing info: asked instead of guessing", 2), ("benign", "Benign, not over-asked / answered", 8)]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True); cols = ["#7f849c", "#a6e3a1", "#7f849c", "#cba6f7"]
for ax, (t, title, n) in zip(axs, types):
    vals = [int(held["by_type"][p][t].split("/")[0]) for p in pipes]
    bars = ax.bar(range(4), [100 * v / n for v in vals], color=cols)
    for i, v in enumerate(vals): ax.text(i, 100 * v / n + 2, f"{v}/{n}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7); ax.set_title(title, fontsize=9); ax.set_ylim(0, 118)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("% correct")
fig.suptitle("Part 2, held-out set (n=15, run once after the code was frozen)", fontsize=10); fig.tight_layout()
fig.savefig(ROOT / "docs" / "img" / "11_part2_heldout.png", dpi=160); print("ok")
