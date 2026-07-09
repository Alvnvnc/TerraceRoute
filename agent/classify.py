"""Klasifikasi kategori task — heuristik regex/keyword (GRATIS, tanpa LLM).

8 kategori resmi Track 1. Klasifikasi menentukan jalur solve & verifikasi.
Tidak perlu sempurna: fungsinya mengarahkan strategi, bukan menilai jawaban.
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

# Kategori "jebakan penalaran": model kecil sering confident-wrong (temuan kalibrasi).
TRAP_CATEGORIES = {LOGICAL, MATH}

_RE_CODE = re.compile(r"```|\bdef \w+\(|\bfunction \w+\(|\breturn\b|\bimport \b|=>|\bclass \w+", re.I)
_RE_DEBUG = re.compile(r"\b(bug|fix|debug|error|wrong|incorrect|broken|corrected?)\b", re.I)
_RE_GEN = re.compile(r"\b(write|implement|create|generate)\b.*\b(function|method|program|code|script)\b", re.I)
_RE_MATH = re.compile(r"\d.*[-+*/%]|\bhow many\b|\bpercent|\b%|\bsum\b|\baverage\b|\btotal\b|"
                      r"\bcalculate\b|\bremain\b|\bcost\b|\bprofit\b|\bmultiply\b|\bdivide\b", re.I)
_RE_SENTIMENT = re.compile(r"\bsentiment\b|\bpositive or negative\b|\bclassify (the )?(sentiment|review|tone)\b", re.I)
_RE_SUMMARY = re.compile(r"\bsummari[sz]e\b|\bsummary\b|\bin (one|two|\d+) sentences?\b|\bcondense\b|\btl;?dr\b", re.I)
_RE_NER = re.compile(r"\bnamed entit|\bextract .*(entit|people|person|organi|location|date)|\bidentify .*(entit|name)", re.I)
_RE_LOGIC = re.compile(r"\bwho owns\b|\bwho is\b.*\?|\bif .* then\b|\beach (own|has)\b|\bpuzzle\b|"
                       r"\bdeduc|\blogical|\bknights? and knaves\b|\bwho (killed|committed)\b", re.I)
_RE_FACT = re.compile(r"\bwhat is\b|\bwho (is|was|were)\b|\bwhen (did|was)\b|\bwhere (is|was)\b|"
                      r"\bexplain\b|\bdefine\b|\bhow (does|do) \b", re.I)


def classify(prompt: str) -> str:
    p = prompt.strip()

    # Kode dulu (paling spesifik)
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
    return FACTUAL  # default aman
