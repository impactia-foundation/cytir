"""Frozen JORF full-text population counts, declared once (task 1437, S7 review fix).

Source: the sealed JORFTEXT identity population (`jorf` = 75,218 in task 469 / task 511 phase B) and
the full-article / source-content-residual split of that population (74,459 + 759, task 516).
A change here is a new sealed run, not an edit.
"""
JORF_FULL_PARENTS = 75_218
JORF_FULL_PARENTS_WITH_CONTENT = 74_459
JORF_FULL_PARENTS_EMPTY = 759

assert JORF_FULL_PARENTS_WITH_CONTENT + JORF_FULL_PARENTS_EMPTY == JORF_FULL_PARENTS
