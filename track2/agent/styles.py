"""Style cards + free deterministic style lint.

The lint is the Track 2 analog of Track 1's verify.py: zero-cost independent checks
that catch the classic failure modes BEFORE the (paid) checker model sees anything.
It never blocks output — a lint failure only triggers one regeneration with feedback.
"""
from __future__ import annotations

import re
from typing import Optional

# The four required styles (missing style = zero for the clip — spec).
FORMAL = "formal"
SARCASTIC = "sarcastic"
HUMOROUS_TECH = "humorous_tech"
HUMOROUS_NON_TECH = "humorous_non_tech"
ALL_STYLES = (FORMAL, SARCASTIC, HUMOROUS_TECH, HUMOROUS_NON_TECH)

# Style cards: definition + do/don't + one exemplar each. Kept compact — they ride on
# every stylize call. Exemplars are about a NEUTRAL subject (a dog on a beach) so the
# model imitates the tone, not the content.
# Shared rule for the humorous styles: joke about the situation or the actions, never
# about a person's body, appearance, or identity (both a safety and a judge-score issue).
_HUMOR_RULE = ("Joke about the situation and actions in the clip — never about a "
               "person's body, appearance, or identity.\n")

CARDS = {
    FORMAL: (
        "Professional, objective, factual tone. Complete sentences, no slang, no emoji, "
        "no exclamation marks, no first person, no jokes.\n"
        'Example: "A golden retriever runs along the shoreline of a sandy beach, '
        'kicking up spray in the late-afternoon light."'
    ),
    SARCASTIC: (
        _HUMOR_RULE +
        "Dry, ironic, lightly mocking — understated wit, never mean or crude. Often "
        "praises something trivial as if it were monumental, or feigns being unimpressed.\n"
        'Example: "Ah yes, a dog running on a beach. Truly the most groundbreaking '
        'footage ever captured."'
    ),
    HUMOROUS_TECH: (
        _HUMOR_RULE +
        "Funny, and it MUST hinge on a technology or programming reference (software, "
        "code, hardware, internet culture) tied to what actually happens in the clip.\n"
        'Example: "Dog.exe has entered turbo mode — sand rendering at 4K, drool physics '
        'fully enabled."'
    ),
    HUMOROUS_NON_TECH: (
        _HUMOR_RULE +
        "Funny, everyday humour anyone would get. STRICTLY no technology, programming, "
        "or internet jargon of any kind.\n"
        'Example: "Somewhere between the third and fourth zoomie, this dog decided the '
        'ocean was his sworn enemy."'
    ),
}

# --- Deterministic lint -------------------------------------------------------------

# Jargon that must NOT appear in humorous_non_tech. Deliberately excludes everyday
# objects (computer, phone, screen): if the clip literally shows one, naming it is
# content, not jargon. This targets tech-culture/programming vocabulary.
_TECH_JARGON = re.compile(
    r"\b(algorithm|software|hardware|firmware|app|apps|code|coding|program(?:ming|mer)?|"
    r"debug\w*|server|cpu|gpu|ram\b|wifi|wi-fi|bluetooth|download\w*|upload\w*|update\w*|"
    r"install\w*|reboot\w*|bug|glitch\w*|pixel\w*|byte\w*|data(?:base)?|internet|online|"
    r"browser|streaming|buffer\w*|render\w*|ai\b|robot\w*|automat\w*|\.exe|404|error "
    r"message|loading|lag(?:gy|ging)?|bandwidth|cloud|crypto\w*|blockchain|meme)\b",
    re.I)

# humorous_tech must contain at least one recognizable tech reference (broader list —
# here everyday devices DO count as a tech reference).
_TECH_REF = re.compile(
    r"\b(algorithm|software|hardware|app|apps|code|coding|program\w*|debug\w*|server|"
    r"cpu|gpu|ram\b|wifi|wi-fi|bluetooth|download\w*|upload\w*|update\w*|install\w*|"
    r"reboot\w*|bug|glitch\w*|pixel\w*|byte\w*|database|internet|online|browser|"
    r"stream\w*|buffer\w*|render\w*|\bai\b|robot\w*|automat\w*|\.exe|404|laptop|"
    r"computer|keyboard|mouse|screen|monitor|phone|smartphone|email|password|login|"
    r"lag(?:gy|ging)?|bandwidth|cloud|version|beta\b|patch\b|firmware|tech|start-?up|"
    r"script\w*|loop\b|load(?:ing|ed)?|wire?less|battery|charging|algorithmic)\b",
    re.I)

_FORMAL_BANNED = re.compile(
    r"(!|\blol\b|\bomg\b|\blmao\b|\bhaha\w*\b|\bkinda\b|\bgonna\b|\bwanna\b|"
    r"\bsuper\b|\btotally\b|\bI\b|\bwe\b|\bour\b|😀|😂|🤣|❤|🔥|✨|🐱|🐶)", re.I)

_EMPTYISH = re.compile(r"^\W*$")


def lint(style: str, caption: str) -> Optional[str]:
    """Return a human-readable problem, or None if the caption passes.

    The message is fed back verbatim into the regeneration prompt.
    """
    text = (caption or "").strip()
    if _EMPTYISH.match(text):
        return "caption is empty"
    if len(text) > 420:
        return "caption is too long — keep it to 1-2 sentences"
    if style == FORMAL and _FORMAL_BANNED.search(text):
        return ("formal tone violated: remove exclamation marks, slang, emoji and "
                "first-person pronouns")
    if style == HUMOROUS_NON_TECH and (m := _TECH_JARGON.search(text)):
        return (f"contains the technical term '{m.group(0)}' — humorous_non_tech must "
                f"have zero technology references; joke about everyday life instead")
    if style == HUMOROUS_TECH and not _TECH_REF.search(text):
        return ("no technology reference found — humorous_tech must hinge on a tech or "
                "programming reference tied to the clip")
    return None
