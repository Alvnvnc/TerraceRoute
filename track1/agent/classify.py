"""Task category classification — regex/keyword heuristics (FREE, no LLM).

The 8 official Track 1 categories. Classification picks the solve & verification path.
It need not be perfect: it steers strategy, it does not grade answers.
"""
from __future__ import annotations

import re

FACTUAL = "factual"
MATH = "math"
SENTIMENT = "sentiment"
SUMMARISATION = "summarisation"
NER = "ner"
CODE_DEBUG = "code_debug"
LOGICAL = "logical"
CODE_GEN = "code_gen"

CATEGORIES = [FACTUAL, MATH, SENTIMENT, SUMMARISATION, NER, CODE_DEBUG, LOGICAL, CODE_GEN]

# Note: the old "trap categories" concept (logical+math) has been superseded by the
# data-driven per-level escalation rungs in solve._LEVEL_CATEGORIES (docs/escalation-math.md).

_RE_CODE = re.compile(r"```|\bdef \w+\(|\bfunction \w+\(|\breturn\b|\bimport \b|=>|\bclass \w+", re.I)
_RE_DEBUG = re.compile(r"\b(bug|fix|debug|error|wrong|incorrect|broken|corrected?)\b", re.I)
_RE_GEN = re.compile(r"\b(write|implement|create|generate)\b.*\b(function|method|program|code|script)\b", re.I)
_RE_MATH = re.compile(r"\d.*[-+*/%]|\bhow many\b|\bpercent|\b%|\bsum\b|\baverage\b|\btotal\b|"
                      r"\bcalculate\b|\bremain\b|\bcost\b|\bprofit\b|\bmultipl|\bdivide[sd]?\b|"
                      r"\bdivisor\b|\bfactorial\b|\bfibonacci\b|\bprime number\b|\bsquare root\b|"
                      r"\bspeed\b|\bkm/h\b", re.I)
_RE_SENTIMENT = re.compile(r"\bsentiment\b|\bpositive or negative\b|\bclassify (the )?(sentiment|review|tone)\b", re.I)
_RE_SUMMARY = re.compile(r"\bsummari[sz]e\b|\bsummary\b|\bin (one|two|\d+) sentences?\b|\bcondense\b|\btl;?dr\b", re.I)
_RE_NER = re.compile(r"\bnamed entit|\bextract .*(entit|people|person|organi|location|date)|\bidentify .*(entit|name)", re.I)
_RE_LOGIC = re.compile(r"\bwho owns\b|\bwho is\b.*\?|\bif .* then\b|\beach (?:\w+ )?(?:has|owns?|is|gets)\b|\bpuzzle\b|"
                       r"\bdeduc|\blogical|\bknights? and knaves\b|\bwho (killed|committed)\b|"
                       # Common riddle/trap markers (eval v4 finding: these often leak into
                       # math/factual but need the trap path): "all but", "are left",
                       # relative dates, syllogisms.
                       r"\ball but\b|\b(?:are|is) left\b|\bnobody (?:leaves|left)\b|"
                       r"\bwhat day\b|\bdays? (?:after|before|from) (?:tomorrow|yesterday|today)\b|"
                       r"\bcan (?:we|you) (?:logically )?conclude\b|\balways (?:tell|lie)|\btruth-?teller|"
                       # Syllogism / deductive inference ("All X are/have Y. ... Does it follow?").
                       # NB "all integers from"/"all numbers from" (math) is NOT matched — the
                       # premise verb (are/have/can) must directly follow the noun.
                       r"\ball [a-z]+ (?:are|have|can)\b|\bdoes it follow\b|\bare all\b|"
                       # Comparison / ordering puzzles.
                       r"\b(?:heavier|lighter|taller|shorter|older|younger|faster|slower|stronger|weaker) than\b|"
                       # Race-position and family-relation riddles (relational, not arithmetic).
                       r"\bovertake\b|\bwhat (?:position|place)\b|"
                       r"\bhow many (?:brothers?|sisters?|sons?|daughters?)\b|"
                       r"\b(?:first|second|third|fourth|fifth|last) (?:daughter|son|child)\b", re.I)
_RE_FACT = re.compile(r"\bwhat is\b|\bwho (is|was|were)\b|\bwhen (did|was)\b|\bwhere (is|was)\b|"
                      r"\bexplain\b|\bdefine\b|\bhow (does|do) \b", re.I)


def classify(prompt: str) -> str:
    p = prompt.strip()

    # Code first (most specific)
    if _RE_CODE.search(p) or _RE_GEN.search(p):
        if _RE_DEBUG.search(p) and _RE_CODE.search(p):
            return CODE_DEBUG
        if _RE_GEN.search(p) or _RE_CODE.search(p):
            return CODE_GEN
    if _RE_SENTIMENT.search(p):
        return SENTIMENT
    if _RE_NER.search(p):
        return NER
    if _RE_SUMMARY.search(p):
        return SUMMARISATION
    if _RE_LOGIC.search(p):
        return LOGICAL
    if _RE_MATH.search(p):
        return MATH
    if _RE_FACT.search(p):
        return FACTUAL
    return FACTUAL  # safe default
