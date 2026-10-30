"""Stage 2a: tools. Math and lookups live in CODE; the LLM only decides when to call them and explains results."""
import json, pathlib, requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}

class ToolError(Exception):
    """Raised for bad arguments; the message is fed back to the model so it can self-correct."""

def _num(args, name, lo, hi):
    v = args.get(name)
    if isinstance(v, str):
        try: v = float(v)
        except ValueError: pass
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ToolError(f"'{name}' must be a number, got {v!r}")
    if not lo <= v <= hi:
        raise ToolError(f"'{name}' must be between {lo} and {hi}, got {v}")
    return float(v)

def search_guidelines(args, retriever):
    q = args.get("query")
    if not isinstance(q, str) or len(q.strip()) < 3:
        raise ToolError("'query' must be a non-empty string")
    hits = retriever.search(q, 3)
    return {"results": [{"source_id": h["id"], "text": h["text"][:700]} for h in hits]}

def calculate_tdee(args, retriever=None):
    kg = _num(args, "weight_kg", 30, 300); cm = _num(args, "height_cm", 120, 230); age = _num(args, "age", 18, 100)
    sex = str(args.get("sex", "")).lower()
    if sex not in ("male", "female"): raise ToolError("'sex' must be 'male' or 'female'")
    act = str(args.get("activity", "")).lower()
    if act not in ACTIVITY: raise ToolError(f"'activity' must be one of {sorted(ACTIVITY)}")
    bmr = 10 * kg + 6.25 * cm - 5 * age + (5 if sex == "male" else -161)   # Mifflin-St Jeor
    return {"bmr_kcal": round(bmr), "tdee_kcal": round(bmr * ACTIVITY[act]),
            "note": "Mifflin-St Jeor estimate; real needs vary by roughly +/-10%."}

def deficit_estimate(args, retriever=None):
    tdee = _num(args, "tdee", 1000, 6000); intake = _num(args, "intake", 0, 8000)
    if intake < 1200:   # tool-level guardrail, independent of the prompt
        raise ToolError("Intake below 1200 kcal/day is outside what this tool will assess; advise seeing a professional.")
    d = tdee - intake
    return {"daily_difference_kcal": round(d), "direction": "deficit" if d > 0 else "surplus" if d < 0 else "maintenance",
            "approx_weekly_change_kg": round(-d * 7 / 7700, 2),
            "note": "Rough rule of thumb (~7,700 kcal per kg of body weight); real-world change is not linear and slows over time."}

_FDC_CACHE = ROOT / "data" / "fdc_cache.json"
def lookup_nutrition(args, retriever=None):
    food = args.get("food")
    if not isinstance(food, str) or len(food.strip()) < 2: raise ToolError("'food' must be a non-empty string")
    cache = json.loads(_FDC_CACHE.read_text()) if _FDC_CACHE.exists() else {}
    key = food.strip().lower()
    if key not in cache:
        r = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", timeout=30,
                         params={"query": key, "pageSize": 1, "dataType": "Foundation,SR Legacy", "api_key": "DEMO_KEY"})
        if r.status_code != 200: raise ToolError(f"USDA lookup failed (HTTP {r.status_code})")
        foods = r.json().get("foods", [])
        if not foods: raise ToolError(f"No USDA entry found for '{food}'")
        f = foods[0]; n = {x["nutrientName"]: x["value"] for x in f.get("foodNutrients", [])}
        cache[key] = {"matched_food": f["description"], "per": "100 g",
                      "kcal": n.get("Energy"), "protein_g": n.get("Protein"),
                      "fat_g": n.get("Total lipid (fat)"), "carbs_g": n.get("Carbohydrate, by difference"),
                      "source": "USDA FoodData Central"}
        _FDC_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]

REGISTRY = {"search_guidelines": search_guidelines, "calculate_tdee": calculate_tdee,
            "deficit_estimate": deficit_estimate, "lookup_nutrition": lookup_nutrition}

def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

SCHEMAS = [
    _fn("search_guidelines", "Search the nutrition guideline documents (NIDDK, MedlinePlus) for educational facts.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("calculate_tdee", "Compute BMR and TDEE (maintenance calories) with Mifflin-St Jeor. Always use this for energy-needs math.",
        {"weight_kg": {"type": "number"}, "height_cm": {"type": "number"}, "age": {"type": "number"},
         "sex": {"type": "string", "enum": ["male", "female"]},
         "activity": {"type": "string", "enum": list(ACTIVITY)}}, ["weight_kg", "height_cm", "age", "sex", "activity"]),
    _fn("deficit_estimate", "Needs the numeric tdee_kcal RETURNED by calculate_tdee (call that first). Given TDEE and daily intake in kcal, return the deficit/surplus and a rough weekly weight change.",
        {"tdee": {"type": "number"}, "intake": {"type": "number"}}, ["tdee", "intake"]),
    _fn("lookup_nutrition", "Look up calories and macros per 100 g of a food in USDA FoodData Central.",
        {"food": {"type": "string"}}, ["food"]),
]


def calculate_tdee_grounded(args, slots):
    """Part 2: refuse to compute from values the user never stated (stops the model inventing age/sex/etc.)."""
    for k in ("weight_kg", "height_cm", "age"):
        v = args.get(k)
        try: v = float(v)
        except (TypeError, ValueError): raise ToolError(f"'{k}' must be a number")
        if slots.get(k) is None or abs(v - slots[k]) > 0.01 * max(slots[k], 1):
            raise ToolError(f"'{k}'={v} was not stated by the user. Do not assume it: ask the user for it.")
    if str(args.get("sex", "")).lower() != slots.get("sex"):
        raise ToolError("'sex' was not stated by the user. Do not assume it: ask the user for it.")
    assumed = slots.get("activity") is None
    a = dict(args); a["activity"] = "sedentary" if assumed else slots["activity"]
    r = calculate_tdee(a)
    if assumed: r["assumption"] = "Activity level was NOT stated; 'sedentary' was assumed. You must tell the user this and that TDEE is higher if they are more active."
    return r
