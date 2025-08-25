# utils/norms/dummy_rect_keepout.py
# --------------------------------------------------------------------
# RectKeepOut — simple spatial rectangular "no-go" zone
# --------------------------------------------------------------------
# PURPOSE:
#   - Hard-blocks a user-defined rectangle of cells.
#   - Purely spatial; no dependence on apples or time.
#
# HOW IT'S USED:
#   - A* treats hard-blocked cells like walls via the norm callback.
#   - You can override `top`, `left`, `bottom`, `right` from the driver.
#
# EXAMPLE (in scripts/run_agents.py):
#   norm = load_norm("dummy_rect_keepout:RectKeepOut")
#   norm.top, norm.left, norm.bottom, norm.right = 5, 5, 8, 8
#
# NOTES:
#   - Coordinates are inclusive (top/left/bottom/right).
#   - We implement the NormSpec interface.
#   - `bind()` and `on_step()` are no-ops here (not needed).
#   - `set_epsilon()` is ignored; this is a hard rule.
# --------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Optional
from utils.norms.base import NormSpec, StateView

Coord = Tuple[int, int]


class RectKeepOut(NormSpec):
    name = "rect_keep_out"

    def __init__(self):
        # Default rectangle (override from driver):
        # Forbid cells (r, c) with top <= r <= bottom AND left <= c <= right
        self.top: int = 5
        self.left: int = 5
        self.bottom: int = 8
        self.right: int = 8

        self._state: Optional[StateView] = None  # not used in this static rule

    # ---- NormSpec interface ----------------------------------------------------

    def bind(self, state: StateView) -> None:
        self._state = state  # not used; kept for interface completeness

    def on_step(self) -> None:
        pass  # static rule, nothing to updat
