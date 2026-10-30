"""Part 2: deterministic slot extraction + intent planning. NO LLM here, so what the user stated is auditable.
Anything not matched stays None, and None means 'ask', never 'guess'."""
import re

_NUM = r"(\d+(?:\.\d+)?)"
LB, FT_CM, IN_CM = 0.45359237, 30.48, 2.54

def _first(pattern, text, flags=re.I):
    return re.search(pattern, text, flags)

def extract(text):
    t = text.replace("’", "'").replace("”", '"').replace("“", '"')
    s = {"weight_kg": None, "height_cm": None, "age": None, "sex": None, "activity": None, "intake": None, "tdee": None}
    m = _first(_NUM + r"\s*(kgs?|kilograms?|kilos?|lbs?|pounds?)\b", t)
    if m: s["weight_kg"] = round(float(m.group(1)) * (1 if m.group(2).lower().startswith("k") else LB), 2)
    if (m := _first(_NUM + r"\s*(?:cm|centimet(?:er|re)s?)\b", t)): s["height_cm"] = float(m.group(1))
    elif (m := _first(r"\b(\d)\s*(?:ft|feet|foot|')\s*(?:(\d{1,2})\s*(?:in(?:ch(?:es)?)?\b|\"|'')?)?", t)):
        s["height_cm"] = round(int(m.group(1)) * FT_CM + int(m.group(2) or 0) * IN_CM, 1)
    elif (m := _first(r"\b([12]\.\d{1,2})\s*m(?:et(?:re|er)s?)?\b", t)): s["height_cm"] = float(m.group(1)) * 100
    for pat in (r"\b(\d{2})\s*[- ]?\s*(?:years?|yrs?)[- ]?\s*old\b", r"\b(\d{2})\s*(?:yo|y/o)\b", r"\bage[d]?\s*:?\s*(\d{2})\b",
                r"\bi(?:'m| am)\s+(\d{2})\b(?!\s*(?:kg|kgs|cm|lbs?|kcal|cal|ft|feet|'|%|g\b))", r"\b(?:female|male|woman|man),?\s*(\d{2})\b"):
        if (m := _first(pat, t)) and 14 <= int(m.group(1)) <= 100:
            s["age"] = int(m.group(1)); break
    fem = _first(r"\b(female|woman|women|girl|lady)\b", t); mal = _first(r"\b(male|man|men|guy|boy)\b", t)
    if fem and not mal: s["sex"] = "female"
    elif mal and not fem: s["sex"] = "male"
    low = t.lower()
    for level, pat in (("very_active", r"very active|extra active|extremely active|athlete|train(?:ing)? twice|6-7 days|hard exercise"),
                       ("light", r"lightly active|light(?:ly)? exercise|1-3 days|light activity"),
                       ("moderate", r"moderately active|moderate (?:exercise|activity)|3-5 days"),
                       ("sedentary", r"sedentary|desk job|little or no exercise|no exercise|inactive|sit all day"),
                       ("active", r"\bactive\b")):
        if re.search(pat, low): s["activity"] = level; break
    def kcal(pat):
        m = _first(pat, t)
        if not m: return None
        v = int(m.group(1).replace(",", ""))
        return v if 500 <= v <= 8000 else None
    s["tdee"] = kcal(r"(?:tdee|maintenance(?: calories)?|burn(?:ing|s)?)\D{0,20}?(\d[\d,]{2,4})")
    s["intake"] = kcal(r"(?:\beat(?:ing|s)?|consum(?:e|ing)|intake(?: is| of)?|take in)\D{0,25}?(\d[\d,]{2,4})")
    return s

TDEE_INTENT = r"\b(tdee|bmr|basal|maintenance|maintain (?:my )?weight|calories? (?:do )?i (?:burn|need)|calorie needs?|daily calorie|energy (?:needs|expenditure)|how many calories should i)\b"
DEFICIT_INTENT = r"(will i (?:lose|gain)|lose or gain|gain or lose|gaining or losing|losing or gaining|how much weight|deficit|surplus|what happens to my weight|weekly (?:weight )?(?:loss|change)|make me (?:lose|gain)|am i (?:gaining|losing))"
FOOD_INTENT = r"usda|fooddata|(?:calories|kcal|protein|carbs?|fat)\s+(?:are\s+)?in\s+(?:\d+\s*(?:g|grams?|oz)\b|a |an |one |1 )"
FACTS_INTENT = r"guideline|recommend|suggest|initial goal|sodium|fiber|fibre|limit|safe pace|how many minutes|what (?:does|do)\b"
PERSONAL = r"\b(i|i'm|im|my|me)\b"

def plan(text, slots=None):
    slots = slots or extract(text); low = text.lower()
    p = {"food": bool(re.search(FOOD_INTENT, low)), "facts": bool(re.search(FACTS_INTENT, low))}
    personal = bool(re.search(PERSONAL, low))
    p["deficit"] = personal and bool(re.search(DEFICIT_INTENT, low))
    p["tdee"] = personal and (bool(re.search(TDEE_INTENT, low)) or (p["deficit"] and slots["tdee"] is None))
    p["calc"] = p["tdee"] or p["deficit"] or p["food"]
    p["need_tdee_calc"] = p["tdee"] and slots["tdee"] is None      # user did not supply a TDEE themselves
    return p

def missing(slots, p):
    miss = []
    if p["need_tdee_calc"] or (p["deficit"] and slots["tdee"] is None):
        miss += [k for k in ("age", "sex", "weight_kg", "height_cm") if slots[k] is None]
    if p["deficit"] and slots["intake"] is None:
        miss.append("intake")
    return miss

LABEL = {"age": "your age", "sex": "your sex (the formula has a male/female term)", "weight_kg": "your weight (kg or lb)",
         "height_cm": "your height (cm or ft/in)", "intake": "how many calories you eat per day"}

def ask_message(miss, slots):
    items = ", ".join(LABEL[k] for k in miss)
    extra = " Activity level is optional: if you skip it I'll assume 'sedentary' and show a range." if slots["activity"] is None and any(k != "intake" for k in miss) else ""
    return (f"I can't safely guess personal details, because they change the result a lot. To answer, I need: {items}. "
            f"Could you share them?{extra}")


def facts_question(text):
    """Keep only the clause(s) that ask for guideline facts, so personal numbers don't pollute retrieval."""
    segs = [x.strip() for x in re.split(r",? and |;|\?|\. ", text) if x.strip()]
    keep = [x for x in segs if re.search(FACTS_INTENT, x.lower())]
    return (" ".join(keep) + "?") if keep else text
