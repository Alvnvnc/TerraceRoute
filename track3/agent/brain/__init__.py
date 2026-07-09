"""The reasoning + safety layer.

``schemas`` defines the typed plan the local models emit (grammar-constrained
JSON) and how to normalize it for comparison. ``gate`` turns a blast-radius
classification and a cross-model disagreement score into an act/ask/refuse
decision — the "knows when not to act" core.
"""
