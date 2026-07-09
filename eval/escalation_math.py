"""Model matematis keputusan eskalasi TerraceRoute.

Sumber data: eval v4 (66 task, grader ketat) di GPU AMD, 10 Jul 2026.
Ketidakpastian dimodelkan Bayesian:
  p_c (akurasi lokal per kategori) ~ Beta(correct+1, wrong+1)
  q   (akurasi remote Gemma-31B-class per task escalated) ~ Beta(37,3)  [mean .925, asumsi]
Hidden set: m=64 task, 8 per kategori (asumsi seimbang; sensitivitas diuji).

Output:
  1. P(lolos gate T) per kebijakan eskalasi, untuk grid T
  2. E[token] per kebijakan (input+output dihitung, sesuai aturan)
  3. Ranking efisiensi marginal per kategori: dAcc/dToken
"""
import json
import random
import statistics

random.seed(7)
N_MC = 20000

# ---- data eval v4 (setelah patch classify) ----
# kategori: (benar, total) grader KETAT
CATS = {
    "ner":           (8, 8),
    "summarisation": (6, 6),
    "code_debug":    (8, 8),
    "code_gen":      (8, 8),
    "sentiment":     (7, 8),
    "factual":       (8, 10),
    "math":          (8, 10),
    "logical":       (7, 8),
}

# estimasi token per task bila DI-ESCALATE (input: prompt+sys remote; output: cap _REMOTE_MAXTOK
# tapi rata2 aktual lebih pendek utk kategori pendek). Dari panjang prompt tasks_v4 (chars/4).
TOK = {  # (input_est, output_est)
    "factual":       (45 + 12, 35),
    "math":          (55 + 14, 12),
    "sentiment":     (40 + 12, 3),
    "summarisation": (95 + 14, 60),
    "ner":           (40 + 14, 40),
    "logical":       (55 + 16, 15),
    "code_debug":    (60 + 14, 120),
    "code_gen":      (50 + 14, 150),
}

M_PER_CAT = 8          # hidden set: task per kategori (asumsi)
Q_A, Q_B = 37, 3       # prior akurasi remote (mean .925)

POLICIES = {
    "L0  zero-API":                      [],
    "P1  math saja":                     ["math"],
    "P2  math+logical (=lvl2 skrg)":     ["math", "logical"],
    "P3  math+factual":                  ["math", "factual"],
    "P4  math+factual+logical":          ["math", "factual", "logical"],
    "P5  semua non-perfect":             ["math", "factual", "logical", "sentiment"],
    "P6  semua 8 kategori (~lvl3)":      list(CATS),
}

T_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92]


def sample_beta(a, b):
    x = random.gammavariate(a, 1.0)
    y = random.gammavariate(b, 1.0)
    return x / (x + y)


def run_policy(escalated: list[str]):
    """Return (list akurasi hidden-set MC, E[token])."""
    accs = []
    etok = sum((TOK[c][0] + TOK[c][1]) * M_PER_CAT for c in escalated)
    for _ in range(N_MC):
        q = sample_beta(Q_A, Q_B)
        n_ok = 0
        n_tot = 0
        for c, (k, n) in CATS.items():
            p = sample_beta(k + 1, n - k + 1)
            eff = q if c in escalated else p     # escalate: jawaban diganti remote
            n_ok += sum(random.random() < eff for _ in range(M_PER_CAT))
            n_tot += M_PER_CAT
        accs.append(n_ok / n_tot)
    return accs, etok


print(f"{'kebijakan':<34}{'E[acc]':>7}{'E[tok]':>8}", end="")
for t in T_GRID:
    print(f"  P≥{t:.2f}", end="")
print()
results = {}
for name, esc in POLICIES.items():
    accs, etok = run_policy(esc)
    mean_acc = statistics.fmean(accs)
    results[name] = (accs, etok)
    print(f"{name:<34}{mean_acc:>7.3f}{etok:>8}", end="")
    for t in T_GRID:
        p = sum(a >= t for a in accs) / len(accs)
        print(f"{p:>8.2f}", end="")
    print()

# ---- efisiensi marginal per kategori (dAcc/dToken), analitik pada mean ----
print("\nEfisiensi marginal eskalasi per kategori (asumsi q=.925):")
print(f"{'kategori':<15}{'p_lokal':>8}{'dAcc/task':>10}{'tok/task':>9}{'efisiensi':>12}")
rows = []
q = Q_A / (Q_A + Q_B)
for c, (k, n) in CATS.items():
    p = (k + 1) / (n + 2)
    dacc = (1 - p) * q - p * (1 - q)      # error diperbaiki − jawaban benar dirusak
    tok = TOK[c][0] + TOK[c][1]
    rows.append((dacc / tok * 1000, c, p, dacc, tok))
for eff, c, p, dacc, tok in sorted(rows, reverse=True):
    print(f"{c:<15}{p:>8.3f}{dacc:>10.3f}{tok:>9}{eff:>10.2f}‰")

# ---- verify-then-fix vs escalate-langsung (per kategori) ----
print("\nVerify-then-fix (remote cek YES/NO jawaban lokal; kalau NO, regenerate):")
print(f"{'kategori':<15}{'E[tok] direct':>14}{'E[tok] verify':>14}{'hemat':>8}")
for c, (k, n) in CATS.items():
    p = (k + 1) / (n + 2)
    tin, tout = TOK[c]
    direct = tin + tout
    # verify: input = prompt + jawaban lokal (≈output) + instruksi(15), output = 1
    verify = (tin + tout + 15 + 1) + (1 - p) * direct
    print(f"{c:<15}{direct:>14.0f}{verify:>14.0f}{direct-verify:>8.0f}")
