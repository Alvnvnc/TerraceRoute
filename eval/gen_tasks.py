"""Generator eval set besar & DETERMINISTIK (jawaban dihitung, dijamin benar).

Motivasi: n=30 terlalu kecil -> estimasi korelasi & threshold tak stabil. Menulis 150 task
manual rawan salah-referensi. Solusi: hasilkan task berjawaban-pasti secara terprogram
(aritmetika, GCD, prima, basis, modular, deret, hari, sudut jam, dll) dengan ground-truth
yang dihitung Python + campur beberapa task terbuka (judge). Banyak sengaja dibuat "jebakan"
yang sering menjatuhkan LLM (perbandingan desimal, hitung huruf, riddle) agar kalibrasi punya
error yang cukup.

Jalankan:  python -m eval.gen_tasks   ->  menulis eval/sample_tasks_v3.jsonl
"""
from __future__ import annotations

import json
import random
from math import gcd
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent / "sample_tasks_v3.jsonl"

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def nth_prime(k: int) -> int:
    primes, c = [], 2
    while len(primes) < k:
        if all(c % p for p in primes if p * p <= c):
            primes.append(c)
        c += 1
    return primes[-1]


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def numans(x) -> list[str]:
    """Varian string jawaban numerik yang bisa diterima grader exact."""
    if isinstance(x, float) and x == int(x):
        x = int(x)
    if isinstance(x, int):
        return [str(x), f"{x:,}"]
    return [str(x)]


TASKS: list[dict] = []
_id = 0


def add(prompt: str, answer, difficulty="medium", grader="exact", reference=None):
    global _id
    _id += 1
    t = {"id": _id, "input": prompt, "grader": grader, "difficulty": difficulty}
    if grader == "exact":
        t["answer"] = answer if isinstance(answer, list) else numans(answer)
    else:
        t["reference"] = reference
    TASKS.append(t)


# --- Aritmetika multi-langkah & persen (verifiable) ---
for _ in range(10):
    a, b = random.randint(12, 99), random.randint(12, 99)
    add(f"What is {a} * {b}? Answer with just the number.", a * b)
for _ in range(8):
    p, n = random.choice([10, 15, 20, 25, 30, 40]), random.choice([40, 60, 80, 120, 200])
    add(f"What is {p}% of {n}? Answer with just the number.", p * n // 100)
for _ in range(8):
    price = random.choice([80, 100, 120, 150, 200])
    d1, d2 = random.choice([10, 20, 25]), random.choice([10, 15, 20])
    final = price * (100 - d1) / 100 * (100 - d2) / 100
    add(f"A product costs ${price}. It is discounted {d1}%, then an additional {d2}% off the "
        f"reduced price. Final price in dollars? Answer with just the number.", round(final, 2),
        difficulty="hard")

# --- Teori bilangan ---
for _ in range(8):
    a, b = random.randint(12, 96), random.randint(12, 96)
    add(f"What is the greatest common divisor of {a} and {b}? Answer with just the number.", gcd(a, b))
for _ in range(6):
    a, b = random.randint(4, 20), random.randint(4, 20)
    add(f"What is the least common multiple of {a} and {b}? Answer with just the number.",
        a * b // gcd(a, b))
for k in [5, 7, 8, 10, 12, 15, 20]:
    add(f"What is the {k}th prime number? Answer with just the number.", nth_prime(k))
for _ in range(6):
    a, b, m = random.randint(2, 6), random.randint(5, 12), random.choice([7, 9, 11, 13])
    add(f"Compute {a}^{b} mod {m}. Answer with just the number.", pow(a, b, m), difficulty="hard")

# --- Basis & representasi ---
for _ in range(6):
    n = random.randint(9, 200)
    add(f"Convert the decimal number {n} to binary. Give only the binary digits.", bin(n)[2:])
for _ in range(5):
    b = format(random.randint(5, 250), "b")
    add(f"What is the binary number {b} in decimal? Answer with just the number.", int(b, 2))
for _ in range(5):
    n = random.randint(20, 400)
    add(f"Convert the decimal number {n} to hexadecimal. Give only the hex digits (no 0x).",
        [format(n, "x"), format(n, "X"), "0x" + format(n, "x")])
for n in [4, 9, 14, 29, 44, 49, 90, 99]:
    add(f"Convert the decimal number {n} to Roman numerals.", [roman(n)])

# --- Deret ---
for _ in range(8):
    a1, d = random.randint(1, 9), random.randint(2, 9)
    seq = [a1 + i * d for i in range(5)]
    add("What is the next number in the sequence: " + ", ".join(map(str, seq)) +
        ", ? Answer with just the number.", a1 + 5 * d)
for _ in range(5):
    a1, r = random.randint(1, 4), random.choice([2, 3])
    seq = [a1 * r ** i for i in range(4)]
    add("What is the next number in the sequence: " + ", ".join(map(str, seq)) +
        ", ? Answer with just the number.", a1 * r ** 4, difficulty="hard")

# --- Hari & waktu ---
for _ in range(8):
    start, n = random.randint(0, 6), random.randint(10, 200)
    add(f"If today is {WEEK[start]}, what day of the week will it be {n} days from now?",
        [WEEK[(start + n) % 7]])
for h, m in [(3, 0), (6, 0), (9, 0), (3, 15), (2, 20), (4, 40)]:
    ang = abs(30 * (h % 12) - 5.5 * m)
    ang = min(ang, 360 - ang)
    add(f"On an analog clock at {h}:{m:02d}, what is the angle in degrees between the hour and "
        f"minute hands? Answer with just the number.", round(ang, 1), difficulty="hard")

# --- Fibonacci / faktorial / jumlah ---
for k in [7, 10, 12, 15]:
    add(f"What is the {k}th Fibonacci number, where fib(1)=1, fib(2)=1? Answer with just the number.",
        fib(k), difficulty="hard")
for n in [5, 6, 7]:
    import math
    add(f"What is {n} factorial ({n}!)? Answer with just the number.", math.factorial(n))
for n in [10, 50, 100]:
    add(f"What is the sum of all integers from 1 to {n}? Answer with just the number.", n * (n + 1) // 2)

# --- Jebakan klasik (sering menjatuhkan LLM) ---
add("Which number is larger: 9.11 or 9.9? Answer with just the number.", ["9.9"])
add("Which number is larger: 8.8 or 8.15? Answer with just the number.", ["8.8"])
add("Which is larger: 0.3 or 0.25? Answer with just the number.", ["0.3"])
add("How many times does the letter 'r' appear in the word 'strawberry'? Answer with just the number.", 3)
add("How many times does the letter 's' appear in 'mississippi'? Answer with just the number.", 4)
add("How many letters are in the word 'onomatopoeia'? Answer with just the number.", 12)
add("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does "
    "the ball cost in dollars? Answer with just the number.", ["0.05", "$0.05", ".05"], "hard")
add("Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? "
    "Answer with just the number.", 1, "hard")
add("There are three killers in a room. Someone enters and kills one of them. Nobody leaves. "
    "How many killers are left? Answer with just the number.", 3, "hard")
add("I have 3 apples today. Yesterday I ate one. How many apples do I have now? "
    "Answer with just the number.", 3)
add("A snail climbs a 10m well: +3m each day, -2m each night. On which day does it reach the top? "
    "Answer with just the number.", 8, "hard")
add("A farmer has 17 sheep. All but 9 run away. How many sheep are left? Answer with just the number.", 9)
add("If it takes 5 machines 5 minutes to make 5 widgets, how many minutes for 100 machines to make "
    "100 widgets? Answer with just the number.", 5, "hard")

# --- Aljabar ringkas ---
for _ in range(6):
    a, b = random.randint(2, 12), random.randint(2, 12)
    add(f"If a={a} and b={b}, what is a^2 + 2ab + b^2? Answer with just the number.", (a + b) ** 2)

# --- Terbuka (judge independen) ---
OPEN = [
    ("Explain what Docker is in two sentences.",
     "Docker packages applications and dependencies into portable containers that run consistently across environments.", "easy"),
    ("Rewrite concisely: 'In light of the fact that it was raining, we decided to stay indoors.'",
     "Because it was raining, we stayed indoors.", "easy"),
    ("Explain the difference between TCP and UDP in three points.",
     "TCP is connection-oriented, reliable/ordered with retransmission; UDP is connectionless, unreliable, faster with lower overhead.", "medium"),
    ("Write a Python function fib(n) returning the nth Fibonacci number iteratively (fib(0)=0, fib(1)=1).",
     "def fib(n):\n a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a", "medium"),
    ("Analyze and give the single most likely root cause: 'psycopg2.OperationalError: FATAL: too many connections for role app_user'.",
     "The Postgres connection limit is exhausted; connections not pooled/released. Fix with pooling and closing connections.", "medium"),
    ("Prove that the sum of the first n odd numbers equals n^2 by induction.",
     "Base n=1: 1=1^2. Step: assume first k odds sum to k^2; the (k+1)th odd is 2k+1, so k^2+2k+1=(k+1)^2. QED.", "hard"),
    ("Design a secure multi-tenant RBAC architecture with dynamic permissions, audit logging, and tenant isolation.",
     "Per-tenant roles/permissions/bindings, a policy decision point, enforcement at API middleware, tenant_id with row-level security, append-only audit log.", "hard"),
    ("Given severe class imbalance (1:1000), compare three strategies to improve minority-class recall without destroying precision.",
     "Resampling (SMOTE/undersampling), class-weighted loss, and threshold/cost-sensitive tuning, discussing the precision-recall tradeoff.", "hard"),
    ("What is the derivative of f(x) = x^3 * ln(x)?",
     "f'(x) = 3x^2 ln(x) + x^2 = x^2(3 ln x + 1).", "hard"),
    ("Summarize in one sentence: mitochondria produce ATP through cellular respiration, powering the cell.",
     "Mitochondria generate the cell's energy (ATP) via cellular respiration.", "easy"),
]
for prompt, ref, diff in OPEN:
    add(prompt, None, difficulty=diff, grader="judge", reference=ref)


def main():
    random.shuffle(TASKS)
    for i, t in enumerate(TASKS, 1):
        t["id"] = i
    with open(OUT, "w") as f:
        for t in TASKS:
            f.write(json.dumps(t) + "\n")
    ex = sum(1 for t in TASKS if t["grader"] == "exact")
    print(f"Wrote {len(TASKS)} tasks -> {OUT}  (exact={ex}, judge={len(TASKS)-ex})")


if __name__ == "__main__":
    main()
