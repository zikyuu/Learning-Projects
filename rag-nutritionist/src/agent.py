"""Stage 2b: a hand-written tool-calling agent loop."""
import json, re
from llm import chat
from retrieve import Retriever
import tools, safety, inputs

SYSTEM = """You are a nutrition EDUCATION assistant (not medical advice; no personalised diet plans).
- NEVER do arithmetic yourself: use calculate_tdee and deficit_estimate for any energy/calorie math.
- Use search_guidelines for factual claims and cite the source_id. Use lookup_nutrition for food nutrient values.
- If the tools return nothing relevant, say "I don't have that in my sources."
- Call tools ONE AT A TIME when a later call needs an earlier result: first calculate_tdee, wait for its result, THEN call deficit_estimate with the real tdee_kcal number.
- If a tool returns an error, fix the arguments or explain the limitation. Keep answers under 150 words."""

def parse_text_tool_call(content):
    """Fallback for small models that print a tool call as JSON text instead of a real tool_call."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except json.JSONDecodeError: return None
    name = obj.get("name"); args = obj.get("parameters") or obj.get("arguments") or {}
    if name in tools.REGISTRY and isinstance(args, dict):
        return [{"function": {"name": name, "arguments": args}}]
    return None

def run_agent(question, retriever, max_iters=8, use_guardrail=True, verbose=False, guard_inputs=False, history=None):
    trace = []
    text = "\n".join((history or []) + [question])
    slots = inputs.extract(text)
    if use_guardrail:
        blocked, reason, msg = safety.check(question)
        if blocked:
            return {"answer": msg, "blocked": reason, "trace": trace, "iterations": 0}
    if guard_inputs:                                    # Part 2: ask before computing, never assume
        p = inputs.plan(text, slots); miss = inputs.missing(slots, p)
        if p["calc"] and miss:
            return {"answer": inputs.ask_message(miss, slots), "blocked": None, "ask": True, "trace": trace, "iterations": 0}
    system = SYSTEM + (f"\n- The user's stated personal values are exactly: {json.dumps({k: v for k, v in slots.items() if v is not None})}. "
                       "Use ONLY these. Never invent age, sex, weight, height or activity. If a tool result has an 'assumption', state it in your answer."
                       if guard_inputs else "")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    nudges = 0
    required = []
    if guard_inputs:                                    # code-side checklist of tools this question MUST use
        _p = inputs.plan(text, slots)
        if _p["need_tdee_calc"]: required.append("calculate_tdee")
        if _p["deficit"] and slots["intake"] is not None: required.append("deficit_estimate")
        if _p["facts"]: required.append("search_guidelines")
        if _p["food"]: required.append("lookup_nutrition")
    req_nudges = 0
    for it in range(1, max_iters + 1):
        reply = chat(msgs, tools=tools.SCHEMAS)
        msgs.append(reply)
        calls = reply.get("tool_calls") or parse_text_tool_call(reply.get("content")) or []
        said = [n for n in tools.REGISTRY if n in (reply.get("content") or "")]
        if not calls and said and nudges < 2:           # model *announced* a tool but didn't call it: nudge, bounded
            nudges += 1
            trace.append({"iter": it, "nudge": said[0]})
            msgs.append({"role": "user", "content": f"Please call the {said[0]} tool now instead of describing it."})
            continue
        done = {t["tool"] for t in trace if "tool" in t and not (isinstance(t.get("result"), dict) and "error" in t["result"])}
        pending = [r for r in required if r not in done]
        if not calls and pending and req_nudges < 3:    # tried to answer before using a required tool
            req_nudges += 1
            trace.append({"iter": it, "nudge": pending[0]})
            msgs.append({"role": "user", "content": f"You must call the {pending[0]} tool (using only the user's stated values) before answering. Call it now."})
            continue
        if not calls:                                   # no tool requested -> final answer
            ans = reply["content"].strip()
            if guard_inputs and any(isinstance(t.get("result"), dict) and "assumption" in t["result"] for t in trace) and not re.search(r"assum", ans, re.I):
                ans += "\n\nNote: your activity level wasn't stated, so 'sedentary' was assumed; TDEE would be higher if you are more active."   # enforced in code, not left to the prompt
            return {"answer": ans, "blocked": None, "trace": trace, "iterations": it}
        for c in calls:                                 # handle several tool calls in one turn
            name, args = c["function"]["name"], c["function"]["arguments"]
            if isinstance(args, str):
                try: args = json.loads(args)
                except json.JSONDecodeError: args = {}
            try:
                if name not in tools.REGISTRY: raise tools.ToolError(f"unknown tool '{name}'")
                if guard_inputs and name == "calculate_tdee": result = tools.calculate_tdee_grounded(args, slots)
                else: result = tools.REGISTRY[name](args, retriever)
            except tools.ToolError as e:
                result = {"error": str(e)}
            except Exception as e:                      # never let a tool crash the loop
                result = {"error": f"tool failed: {type(e).__name__}: {e}"}
            trace.append({"iter": it, "tool": name, "args": args, "result": result})
            if verbose: print(f"  [{it}] {name}({args}) -> {str(result)[:150]}")
            msgs.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return {"answer": "I couldn't finish within the step limit.", "blocked": None, "trace": trace, "iterations": max_iters}

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "I'm a 30 year old woman, 65 kg, 165 cm, moderately active. If I eat 2000 kcal will I lose weight?"
    out = run_agent(q, Retriever(), verbose=True)
    print("\nQ:", q, "\nA:", out["answer"], f"\n({out['iterations']} iterations)")
