# utils/norms/dummy_col_band.py
# --------------------------------------------------------------------
# MiddleColumnBandForbidden — simple spatial "keep-out" rule (columns)
# --------------------------------------------------------------------
# PURPOSE:
#   - Hard-blocks a vertical band of columns (e.g., "middle strip is forbidden").
#   - Purely spatial; no dependence on apples or time.
#
# HOW IT'S USED:
#   - A* treats hard-blocked cells like walls via the norm callback.
#   - You can override `col_min` / `col_max` from the driver after loading.
#
# EXAMPLE (in scripts/run_agents.py):
#   norm = load_norm("dummy_col_band:MiddleColumnBandForbidden")
#   norm.col_min = 7
#   norm.col_max = 8
#
# NOTES:
#   - We implement the NormSpec interface.
#   - `bind()` and `on_step()` are no-ops here (not needed).
#   - `set_epsilon()` is ignored; this is a hard rule.
# --------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Optional, Dict, Set
from utils.norms.base import NormSpec, StateView

Coord = Tuple[int, int]


class MiddleColumnBandForbidden(NormSpec):
    name = "middle_column_band_forbidden"

    def __init__(self):
        # Default band (override from driver to fit your map):
        # Forbidden if col_min <= c <= col_max
        self.col_min: int = 6
        self.col_max: int = 7

        # We accept a StateView in bind(), but this norm doesn't need it.
        self._state: Optional[StateView] = None

    # ---- NormSpec interface ----------------------------------------------------

    def bind(self, state: StateView) -> None:
        self._state = state  # not used; kept for interface completeness

    def on_step(self) -> None:
        pass  # no per-step bookkeeping

    def set_epsilon(self, agent_id: str, eps: float) -> None:
        pass  # hard rule: epsilon ignored intentionally

    def hard_blocked(self, agent_id: str, cell: Coord) -> bool:
        r, c = cell
        return self.col_min <= int(c) <= self.col_max

    def soft_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        return 0.0  # no soft costs in this dummy norm
