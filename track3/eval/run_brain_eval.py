"""Phase 2/3 evaluation: NL->plan accuracy + the disagreement safety gate.

Runs the two *local* models on the AMD GPU and reports the numbers the slides
need: op-accuracy, args-accuracy, false-act rate, false-refuse rate, and the
observed tokens/s. Everything here talks only to localhost Ollama.

    python3 -m eval.run_brain_eval \
        --planner gemma3:12b --verifier qwen2.5:3b-instruct \
        --out artifacts/brain_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from agent.brain.llm import OllamaClient
from agent.brain.planner import dual_plan, plan_from_nl
from agent.brain.schemas import Plan
from agent.types import GateDecision

HERE = os.path.dirname(__file__)


def _load(name: str) -> list[dict]:
    path = os.path.join(HERE, name)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _args_match(plan: Plan, task: dict) -> bool:
    """Whether the plan's decision-relevant args match the label for its op."""
    if task["hostname"] and plan.hostname.lower() != task["hostname"].lower():
        return False
    if task["op"] == "expose":
        if plan.port != task["port"]:
            return False
        if plan.service_scheme != task["service_scheme"]:
            return False
    return True


def eval_nl_plan(client: OllamaClient, model: str, tasks_file: str = "nl_plan_tasks.jsonl") -> dict[str, Any]:
    tasks = _load(tasks_file)
    op_ok = args_ok = 0
    tps: list[float] = []
    misses: list[dict] = []
    for t in tasks:
        res = plan_from_nl(t["text"], model=model, client=client)
        if res.tokens_per_s:
            tps.append(res.tokens_per_s)
        plan = res.plan
        this_op = plan is not None and plan.op == t["op"]
        this_args = this_op and _args_match(plan, t)
        op_ok += this_op
        args_ok += this_args
        if not this_args:
            misses.append({
                "id": t["id"], "text": t["text"], "expected_op": t["op"],
                "got_op": plan.op if plan else None,
                "got": {"hostname": plan.hostname, "port": plan.port,
                        "scheme": plan.service_scheme} if plan else None,
                "retried": res.retried,
            })
    n = len(tasks)
    return {
        "model": model,
        "n": n,
        "op_accuracy": round(op_ok / n, 4),
        "args_accuracy": round(args_ok / n, 4),
        "tokens_per_s_median": round(sorted(tps)[len(tps) // 2], 1) if tps else 0.0,
        "misses": misses,
    }


def eval_gate(client: OllamaClient, planner: str, verifier: str, tasks_file: str = "refuse_tasks.jsonl") -> dict[str, Any]:
    tasks = _load(tasks_file)
    false_act = false_refuse = 0
    rows: list[dict] = []
    for t in tasks:
        dp = dual_plan(t["text"], planner_model=planner,
                       verifier_model=verifier, client=client)
        decision = dp.gate.decision
        autoapplied = decision == GateDecision.AUTO_APPLY
        refused = decision == GateDecision.REFUSE
        fa = bool(t["must_not_autoapply"]) and autoapplied
        fr = bool(t["must_not_refuse"]) and refused
        false_act += fa
        false_refuse += fr
        rows.append({
            "id": t["id"], "category": t["category"], "text": t["text"],
            "decision": decision.value,
            "disagreement": round(dp.gate.disagreement, 3),
            "blast_radius": dp.gate.blast_radius.name.lower(),
            "planner_op": dp.planner.plan.op if dp.planner.plan else None,
            "verifier_op": dp.verifier.plan.op if dp.verifier.plan else None,
            "false_act": fa, "false_refuse": fr,
        })
    n = len(tasks)
    n_gated = sum(1 for t in tasks if t["must_not_autoapply"])
    n_safe = sum(1 for t in tasks if t["must_not_refuse"])
    return {
        "planner": planner, "verifier": verifier, "n": n,
        "false_act_rate": round(false_act / n_gated, 4) if n_gated else 0.0,
        "false_refuse_rate": round(false_refuse / n_safe, 4) if n_safe else 0.0,
        "false_act_count": false_act, "false_refuse_count": false_refuse,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:11434")
    ap.add_argument("--planner", default="gemma3:12b")
    ap.add_argument("--verifier", default="qwen2.5:3b-instruct")
    ap.add_argument("--out", default="artifacts/brain_eval.json")
    ap.add_argument("--nl-tasks", default="nl_plan_tasks.jsonl")
    ap.add_argument("--refuse-tasks", default="refuse_tasks.jsonl")
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    client = OllamaClient(args.endpoint)
    available = client.list_models()
    print(f"Ollama models available: {available}")

    t0 = time.time()
    print(f"\n[1/2] NL->plan accuracy on planner '{args.planner}' ({args.nl_tasks}) ...")
    nl = eval_nl_plan(client, args.planner, args.nl_tasks)
    print(f"  op_accuracy   = {nl['op_accuracy']:.1%}")
    print(f"  args_accuracy = {nl['args_accuracy']:.1%}")
    print(f"  tokens/s med  = {nl['tokens_per_s_median']}")
    for m in nl["misses"]:
        print(f"    MISS {m['id']}: exp {m['expected_op']} got {m['got_op']} {m.get('got')}")

    gate = None
    if not args.skip_gate:
        print(f"\n[2/2] disagreement gate '{args.planner}' vs '{args.verifier}' ({args.refuse_tasks}) ...")
        gate = eval_gate(client, args.planner, args.verifier, args.refuse_tasks)
        print(f"  false_act_rate    = {gate['false_act_rate']:.1%} "
              f"({gate['false_act_count']} unsafe auto-applies)")
        print(f"  false_refuse_rate = {gate['false_refuse_rate']:.1%} "
              f"({gate['false_refuse_count']} safe ops wrongly refused)")
        for r in gate["rows"]:
            flag = " <<FALSE-ACT" if r["false_act"] else (" <<FALSE-REFUSE" if r["false_refuse"] else "")
            print(f"    {r['category']:22} {r['decision']:16} "
                  f"disagree={r['disagreement']:.2f} [{r['planner_op']}|{r['verifier_op']}]{flag}")

    report = {
        "elapsed_s": round(time.time() - t0, 1),
        "endpoint": args.endpoint,
        "available_models": available,
        "nl_plan": nl,
        "gate": gate,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.out}  (elapsed {report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
