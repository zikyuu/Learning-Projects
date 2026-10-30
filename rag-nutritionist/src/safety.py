"""Guardrails: rule-based pre-check that runs BEFORE any retrieval/LLM call.
Deliberately conservative: over-refusing on a health topic is cheaper than under-refusing."""
import re

REDIRECT = ("I can't help with that one. I'm a general nutrition-education tool, not medical advice, and I don't "
            "give personalised diet plans. {why} Please talk to a doctor, a registered dietitian, or a qualified "
            "health professional. If you're in crisis or feel unsafe, contact local emergency services.")

RULES = [
    ("very-low-calorie target", r"\b(?:eat|eating|consume|intake|only|under|below|just)\D{0,20}\b([1-9]\d{2}|1[01]\d{2})\s*(?:kcal|calories|cals?)\b|\b([1-9]\d{2}|1[01]\d{2})\s*(?:kcal|calories|cals?)\s*(?:a|per|/)\s*day\b|\bvery low calorie\b|\bcrash diet\b|\bstarv(?:e|ing)\b|\bfast(?:ing)? for (?:\d+|several) days\b",
     "Very low calorie targets can be unsafe without supervision."),
    ("disordered eating", r"\b(?:purg\w*|binge\w*|make myself (?:throw|vomit)|throw(?:ing)? up (?:after|my food)|laxatives? to lose|anorex\w*|bulimi\w*|afraid of (?:eating|food)|hate my body|obsess\w* (?:over|about) (?:food|calories|weight))\b",
     "What you describe can be a sign of disordered eating, which needs professional support."),
    ("unsafe restriction symptoms", r"\b(?:skip(?:ping|ped)? meals|not eating (?:for|in) (?:days|a week)|(?:feel|feeling) (?:dizzy|faint|weak)|faint(?:ing)?)\b",
     "Skipping meals with symptoms like dizziness can be a sign of harmful restriction."),
    ("pregnancy / breastfeeding", r"\b(?:pregnan\w*|breastfeed\w*|nursing mother|lactating|trying to conceive)\b",
     "Nutrition needs change during pregnancy and breastfeeding."),
    ("medical condition / medication", r"\b(?:diabet\w*|kidney (?:disease|failure)|ckd|heart (?:disease|failure)|cancer|insulin|metformin|ozempic|semaglutide|thyroid|celiac|crohn\w*|eating disorder|my doctor|medication|medicine)\b",
     "Medical conditions and medications change what is safe to eat."),
    ("minor", r"\b(?:i am|i'm|im)\s*(?:1[0-7]|[5-9])\s*(?:years? old|yo)\b|\bmy (?:\d{1,2}[- ]?(?:years?[- ]?old|yo)[- ]?)?(?:son|daughter|child|kid)\b|\b\d{1,2}[- ]?years?[- ]?old (?:son|daughter|child|kid|boy|girl)\b",
     "Children and teens have different nutritional needs."),
]

def check(text):
    """Return (blocked: bool, reason: str|None, redirect_message: str|None)."""
    t = text.lower()
    for name, pattern, why in RULES:
        if re.search(pattern, t):
            return True, name, REDIRECT.format(why=why)
    return False, None, None
