"""Write publishable copies of the run files into results/public/ with retrieved page text removed.
The full files (results/eval_runs*.json, part2_runs.json) stay local and are gitignored, since they
contain excerpts of the source pages whose licences I haven't confirmed for redistribution."""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "public"
FILES = ["eval_runs", "eval_runs_v1", "part2_runs", "part2_two_turn"]

def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "text"}   # 'text' is only ever retrieved page text here
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in FILES:
        d = strip(json.loads((ROOT / "results" / f"{f}.json").read_text()))
        (OUT / f"{f}.json").write_text(json.dumps(d, indent=1))
        print("wrote", OUT / f"{f}.json")
