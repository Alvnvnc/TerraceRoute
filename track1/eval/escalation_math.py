"""TerraceRoute escalation decision model.

Data source: eval v4 (66 tasks, strict grader) on an AMD GPU.
Uncertainty is modelled in a Bayesian way:
  p_c (local accuracy per category) ~ Beta(correct+1, wrong+1)
  q   (remote Gemma-31B-class accuracy per escalated task) ~ Beta(37,3)  [mean .925, assumed]
Hidden set: m=64 tasks, 8 per category (balanced assumption; sensitivity is tested).

Output:
  1. P(pass gate T) per escalation policy, over a grid of T
  2. E[tokens] per policy (input+output counted, per the rules)
  3. Marginal-efficiency ranking per category: dAcc/dToken
"""
import random
import statistics

random.seed(7)
N_MC = 20000

# ---- eval v4 data (after the classify patch) ----
# category: (correct, total) under the STRICT grader
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

# Estimated tokens per task IF ESCALATED (input: prompt + remote system; output: the
# _REMOTE_MAXTOK cap, though the actual average is shorter for short categories). Derived
# from the tasks_v4 prompt lengths (chars/4).
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

M_PER_CAT = 8          # hidden set: tasks per category (assumption)
Q_A, Q_B = 37, 3       # remote accuracy prior (mean .925)

POLICIES = {
    "L0  zero-API":                      [],
    "P1  math only":                     ["math"],
    "P2  math+logical (=old lvl2)":      ["math", "logical"],
    "P3  math+factual":                  ["math", "factual"],
    "P4  math+factual+logical":          ["math", "factual", "logical"],
    "P5  all non-perfect":               ["math", "factual", "logical", "sentiment"],
    "P6  all 8 categories (~lvl4)":      list(CATS),
}

T_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92]


def sample_beta(a, b):
    x = random.gammavariate(a, 1.0)
    y = random.gammavariate(b, 1.0)
    return x / (x + y)


def run_policy(escalated: list[str]):
    """Return (list of MC hidden-set accuracies, E[tokens])."""
    accs = []
    etok = sum((TOK[c][0] + TOK[c][1]) * M_PER_CAT for c in escalated)
    for _ in range(N_MC):
        q = sample_beta(Q_A, Q_B)
        n_ok = 0
        n_tot = 0
        for c, (k, n) in CATS.items():
            p = sample_beta(k + 1, n - k + 1)
            eff = q if c in escalated else p     # escalate: the answer is replaced by remote
            n_ok += sum(random.random() < eff for _ in range(M_PER_CAT))
            n_tot += M_PER_CAT
        accs.append(n_ok / n_tot)
    return accs, etok


print(f"{'policy':<34}{'E[acc]':>7}{'E[tok]':>8}", end="")
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

# ---- marginal efficiency per category (dAcc/dToken), analytic at the mean ----
print("\nMarginal escalation efficiency per category (assuming q=.925):")
print(f"{'category':<15}{'p_local':>8}{'dAcc/task':>10}{'tok/task':>9}{'efficiency':>12}")
rows = []
q = Q_A / (Q_A + Q_B)
for c, (k, n) in CATS.items():
    p = (k + 1) / (n + 2)
    dacc = (1 - p) * q - p * (1 - q)      # errors fixed − correct answers broken
    tok = TOK[c][0] + TOK[c][1]
    rows.append((dacc / tok * 1000, c, p, dacc, tok))
for eff, c, p, dacc, tok in sorted(rows, reverse=True):
    print(f"{c:<15}{p:>8.3f}{dacc:>10.3f}{tok:>9}{eff:>10.2f}‰")

# ---- verify-then-fix vs escalate-directly (per category) ----
print("\nVerify-then-fix (remote checks the local answer YES/NO; regenerate on NO):")
print(f"{'category':<15}{'E[tok] direct':>14}{'E[tok] verify':>14}{'saving':>8}")
for c, (k, n) in CATS.items():
    p = (k + 1) / (n + 2)
    tin, tout = TOK[c]
    direct = tin + tout
    # verify: input = prompt + local answer (≈output) + instruction(15), output = 1
    verify = (tin + tout + 15 + 1) + (1 - p) * direct
    print(f"{c:<15}{direct:>14.0f}{verify:>14.0f}{direct-verify:>8.0f}")
