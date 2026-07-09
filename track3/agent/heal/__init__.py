"""Deterministic self-healing: signal taxonomy + watchdog.

Diagnosis is 100% rules (``taxonomy.diagnose``) so it is reliable on stage and
unit-testable without a live tunnel. The LLM only narrates the result and takes
over the ``UNKNOWN`` tail.
"""
