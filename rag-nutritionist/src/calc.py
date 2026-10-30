"""Part 2: run the calculators in code and compose the numeric part of the answer (no LLM arithmetic)."""
import re
import tools, safety

def compute(slots, p, question=""):
    """Returns (text, meta). Raises tools.ToolError for out-of-range inputs."""
    lines, meta = [], {"assumed": []}
    tdee = slots["tdee"]
    if p["need_tdee_calc"]:
        act = slots["activity"]
        if act is None:
            act = "sedentary"; meta["assumed"].append("activity=sedentary")
        base = dict(weight_kg=slots["weight_kg"], height_cm=slots["height_cm"], age=slots["age"], sex=slots["sex"])
        r = tools.calculate_tdee({**base, "activity": act}); tdee = r["tdee_kcal"]
        stated = "stated" if not meta["assumed"] else "ASSUMED, not stated by you"
        lines.append(f"Estimated BMR: {r['bmr_kcal']} kcal/day. Estimated TDEE (maintenance): {tdee} kcal/day, using activity level '{act}' ({stated}).")
        if meta["assumed"]:
            rng = {lvl: tools.calculate_tdee({**base, "activity": lvl})["tdee_kcal"] for lvl in ("sedentary", "light", "moderate", "active")}
            lines.append("You didn't say how active you are, so I assumed sedentary. TDEE by activity level: " +
                         ", ".join(f"{k} {v}" for k, v in rng.items()) + " kcal/day.")
            meta["range"] = rng
    elif tdee is not None:
        lines.append(f"Using the TDEE you gave: {tdee} kcal/day.")
    if p["deficit"] and slots["intake"] is not None and tdee is not None:
        d = tools.deficit_estimate({"tdee": tdee, "intake": slots["intake"]})
        diff, wk = d["daily_difference_kcal"], d["approx_weekly_change_kg"]
        if diff > 0:   lines.append(f"Eating {slots['intake']} kcal/day is a deficit of about {diff} kcal/day, roughly {abs(wk):.2f} kg lost per week.")
        elif diff < 0: lines.append(f"Eating {slots['intake']} kcal/day is a surplus of about {-diff} kcal/day, roughly {abs(wk):.2f} kg gained per week.")
        else:          lines.append(f"Eating {slots['intake']} kcal/day is about maintenance.")
        lines.append("(Rule of thumb ~7,700 kcal per kg; real change is not linear and slows over time.)")
        if meta.get("range"):
            dirs = {k: ("deficit" if v - slots["intake"] > 0 else "surplus" if v - slots["intake"] < 0 else "maintenance") for k, v in meta["range"].items()}
            if len(set(dirs.values())) > 1:
                lines.append("Note: the direction depends on your activity level: " + ", ".join(f"{k}: {v}" for k, v in dirs.items()) + ".")
    if p["tdee"] and not p["deficit"] and re.search(r"how many calories should i", question, re.I):
        lines.append("I can estimate maintenance, but I can't set a personal calorie target. A registered dietitian can do that safely.")
    lines.append("These are Mifflin-St Jeor estimates (roughly +/-10%), for education only, not medical advice.")
    return "\n".join(lines), meta

def refusal_for_low_intake(e):
    return safety.REDIRECT.format(why=str(e))
