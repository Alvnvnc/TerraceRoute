"""TerraceRoute escalation decision model.

Default data source: eval v4 (66 tasks, strict grader) on an AMD GPU. Pass
`--results eval/agent_eval_results.jsonl` to recompute the whole ladder from a *measured* run
(e.g. the expanded v5 set) instead of the baked-in v4 numbers — this is how the ladder stays
honest as the eval set grows.

Uncertainty is modelled in a Bayesian way:
  p_c (local accuracy per category) ~ Beta(correct+1, wrong+1)
  q   (remote Gemma-31B-class accuracy per escalated task) ~ Beta(37,3)  [mean .925, assumed]
Hidden set: m tasks per category (balanced assumption; sensitivity is tested).

Output:
  1. P(pass gate T) per escalation policy, over a grid of T
  2. E[tokens] per policy (input+output counted, per the rules)
  3. Marginal-efficiency ranking per category: dAcc/dToken

Run:
  python -m eval.escalation_math                                   # baked-in v4 data
  python -m eval.escalation_math --results eval/agent_eval_results.jsonl   # measured data
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

random.seed(7)
N_MC = 20000

# ---- eval v4 data (after the classify patch) — used unless --results is given ----
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


def load_results(path: str):
    """Recompute CATS + per-category input-token estimate from a measured agent_eval run.

    The results JSONL (one row per task) carries `category`, `correct`, and `prompt`. We derive
    per-category (correct, total) directly, and re-estimate the INPUT token cost from the actual
    prompt lengths (chars/4 + a fixed remote-system overhead). Output-token estimates stay from
    the baked-in TOK map, since a zero-API run produces no remote output to measure.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # cat -> [correct, total]
    chars: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        c = r["category"]
        agg[c][0] += int(bool(r["correct"]))
        agg[c][1] += 1
        chars[c].append(len(r.get("prompt", "")))
    cats = {c: (v[0], v[1]) for c, v in agg.items()}
    tok = {}
    for c in cats:
        in_est = round(statistics.fmean(chars[c]) / 4) + 14     # prompt/4 + remote system prompt
        out_est = TOK.get(c, (0, 30))[1]                        # keep measured/priored output cap
        tok[c] = (in_est, out_est)
    m_per_cat = round(statistics.fmean([n for _, n in cats.values()]))
    return cats, tok, m_per_cat


def report():
    """Print the full decision-model report using the module-level CATS / TOK / M_PER_CAT."""
    print(f"{'policy':<34}{'E[acc]':>7}{'E[tok]':>8}", end="")
    for t in T_GRID:
        print(f"  P≥{t:.2f}", end="")
    print()
    for name, esc in POLICIES.items():
        esc = [c for c in esc if c in CATS]          # tolerate category sets that differ from data
        accs, etok = run_policy(esc)
        mean_acc = statistics.fmean(accs)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="agent_eval_results.jsonl to recompute the ladder from "
                                       "measured data (else baked-in v4 numbers)")
    args = ap.parse_args()
    if args.results:
        if not Path(args.results).exists():
            raise SystemExit(f"results file not found: {args.results}")
        CATS, TOK, M_PER_CAT = load_results(args.results)
        print(f"[data-driven from {args.results}: "
              f"{sum(n for _, n in CATS.values())} tasks, {len(CATS)} categories, "
              f"M_PER_CAT={M_PER_CAT}]\n")
    report()
