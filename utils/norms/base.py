"""
norms/base.py
--------------
This file defines the *minimal, stable* interface that *every* norm must implement.
It is deliberately tiny and read-only (norms cannot mutate the environment).
Your LLM will generate a class that implements `NormSpec`.

Key ideas:
- Norms get a *read-only* view of the world (StateView).
- Norms can enforce constraints in two ways:
  1) HARD: "You simply cannot step on this cell." (treated as an obstacle by A*)
  2) SOFT: "You may step here, but it costs extra." (added to path cost by A*)
- Optional ε-compliance per agent lets us sweep “ignore the norm with probability ε”.

We DO NOT import your environment/agents here. That keeps coupling near-zero.
"""

from __future__ import annotations
from typing import Protocol, Tuple, Dict, Set

Coord = Tuple[int, int]


class StateView(Protocol):
    """
    Read-only queries a norm may need.
    The runner will provide a concrete implementation and update it each tick.
    """

    # --- Static-ish world info ---
    def grid_size(self) -> Tuple[int, int]:
        """Return (rows, cols)."""

    def walls(self) -> Set[Coord]:
        """Set of impassable cells (environment walls/trees/etc.), if applicable."""

    # --- Dynamic state (updated every step by the runner) ---
    def timestep(self) -> int:
        """Current simulation step (0-based)."""

    def apples(self) -> Set[Coord]:
        """Coordinates of *currently present* apples (uneaten)."""

    def agents(self) -> Dict[str, Coord]:
        """Mapping {agent_id: (row, col)} for current positions."""


class NormSpec(Protocol):
    """
    Every norm must implement this interface.

    Lifecycle:
      1) `bind(state)` is called once per episode to give access to StateView.
      2) At each step, runner will call:
         - `on_step()` (optional hook; no side-effects)
         - A* will consult:
             * `hard_blocked(agent_id, cell)` for obstacle-style constraints
             * `soft_penalty(agent_id, cur, nxt)` to adjust step cost

    ε-compliance:
      - `set_epsilon(agent_id, eps)` is invoked by the runner before/at runtime.
      - The norm may ignore this (no-op) if it has no stochastic compliance.
    """

    # Human-friendly name (helpful in logs/plots)
    name: str

    def bind(self, state: StateView) -> None:
        """Called once at episode start. Do *not* store mutable global state elsewhere."""

    def on_step(self) -> None:
        """
        Optional per-step hook—must not mutate the environment or agents.
        Useful for norms that depend on time windows, phases, or rolling conditions.
        """

    def hard_blocked(self, agent_id: str, cell: Coord) -> bool:
        """
        Return True if `cell` is forbidden for `agent_id`. A* treats it as an unwalkable obstacle.
        Keep this fast and deterministic; it runs inside the planner loop.
        """

    def soft_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        """
        Return a non-negative extra cost for stepping from `cur` to `nxt`.
        0.0 means no extra penalty. This is added to the step cost in A*.
        """

    def set_epsilon(self, agent_id: str, eps: float) -> None:
        """
        Probability that `agent_id` ignores soft penalties (and/or hard blocks) on a step.
        A norm may implement this as a no-op if not applicable.
        """
        ...
