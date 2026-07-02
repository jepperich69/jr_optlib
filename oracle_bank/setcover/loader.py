# -*- coding: utf-8 -*-
"""
Loader for OR-Library set-cover (SCP) instances.

Format (Beasley OR-Library):
    line 1:            m n           (rows, columns)
    next tokens:       n column costs
    then for each row i in 1..m:
        k_i            number of columns that cover row i
        k_i column indices (1-based)

Returns 0-based columns. ``covers[j]`` = sorted rows covered by column j.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SetCoverInstance:
    n_rows: int
    n_cols: int
    costs: List[float]
    covers: List[List[int]]   # covers[j] = rows (0-based) covered by column j
    name: str = ""


def load_scp(path) -> SetCoverInstance:
    path = Path(path)
    tokens = path.read_text().split()
    it = iter(tokens)
    m = int(next(it)); n = int(next(it))
    costs = [float(next(it)) for _ in range(n)]
    covers: List[List[int]] = [[] for _ in range(n)]
    for i in range(m):
        k = int(next(it))
        for _ in range(k):
            j = int(next(it)) - 1     # to 0-based
            covers[j].append(i)
    return SetCoverInstance(n_rows=m, n_cols=n, costs=costs,
                            covers=covers, name=path.stem)


def rows_by_column_to_columns_by_row(inst: SetCoverInstance) -> List[List[int]]:
    """Invert covers: for each row, which columns can cover it (for greedy)."""
    by_row: List[List[int]] = [[] for _ in range(inst.n_rows)]
    for j, rows in enumerate(inst.covers):
        for i in rows:
            by_row[i].append(j)
    return by_row
