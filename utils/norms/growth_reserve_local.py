# utils/norms/growth_reserve_local.py
from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
import random

# Import the stable interface types we defined earlier
from utils.norms.base import NormSpec, StateView

Coord = Tuple[int, int]


class GrowthReserveLocal(NormSpec):
    """
    GrowthReserveLocal (soft local-reserve norm)
    --------------------------------------------
    Intuition:
      - Around each apple tile, look within an L2 (Euclidean) radius R (default 2.0).
      - If eating that apple would leave fewer than K other apples in that neighborhood,
        we treat it as *risky for regrowth* and apply a *soft penalty* to that move.
      - Agents can still eat it (it's *not* a hard wall), but it becomes less attractive.
      - ε-compliance per agent: with probability ε, the agent ignores the penalty on that step.

    Why this shape?
      - It encodes a conservation rule: "don't collapse local clusters below K".
      - It's spatial, local, and easy for an LLM to implement/refine (R, K, penalty).

    Interface notes:
      - This class implements the NormSpec protocol:
          * bind(state)            : receive a read-only StateView at episode start.
          * on_step()              : (optional) called every environment tick.
          * hard_blocked(...)      : (not used here) always False — this is a soft-only norm.
          * soft_penalty(...)      : returns extra path cost for cur->nxt moves (0 if none).
          * set_epsilon(agent, ε)  : per-agent probability to ignore the penalty.

      - For backwards compatibility with your current code during migration, we ALSO expose:
          * update_apples(apples)  : old API to set apples each tick (runner-style)
          * is_blocked(...)        : alias to hard_blocked(...)
          * step_penalty(...)      : alias to soft_penalty(...)
          * expected_step_penalty(...): deterministic expected cost = (1-ε)*penalty when breach

        These shims mean you can switch your planner later without breaking current runs.
    """

    # Human-friendly name (helpful in logs/plots)
    name = "growth_reserve_local"

    def __init__(
        self,
        radius: float = 2.0,      # L2 radius to consider neighbors around a candidate apple
        K: int = 3,               # minimum neighbors we want to remain after harvesting
        penalty: float = 5.0,     # extra path cost when a move would breach the K-reserve
        seed: int = 0             # base seed so ε-compliance is reproducible per agent
    ):
        # --- Hyperparameters / fixed config ---
        self.radius = float(radius)
        self.radius_sq = self.radius * self.radius
        self.K = int(K)
        self.penalty = float(penalty)
        self._base_seed = int(seed) & 0xFFFFFFFF

        # --- Runtime state (filled at bind + updated each step) ---
        self._state: Optional[StateView] = None  # set by bind(...)
        self._apples_override: Optional[Set[Coord]] = None
        # ^ if you call update_apples(...) we use that set instead of state.apples()

        # --- Per-agent ε + RNG (for stochastic compliance) ---
        self._eps: Dict[str, float] = {}               # {agent_id: ε}
        self._rng_by_agent: Dict[str, random.Random] = {}  # stable per-agent RNGs

    # ─────────────────────────────────────────────────────────────────────────────
    # NormSpec interface
    # ─────────────────────────────────────────────────────────────────────────────

    def bind(self, state: StateView) -> None:
        """
        Called once at the start of an episode.
        We store the read-only StateView so we can query apples/agents/walls/timestep.
        """
        self._state = state
        self._apples_override = None  # reset any manual override on new episode

    def on_step(self) -> None:
        """
        Optional hook invoked once per environment tick.
        We keep it empty: this norm is purely reactive to the current apple set.
        """
        # Nothing to do; apple set is read on demand each call.
        pass

    def set_epsilon(self, agent_id: str, eps: float) -> None:
        """
        Configure ε-compliance for an agent:
        - With probability ε, that agent ignores the soft penalty on a given step.
        - ε is clamped to [0, 1].
        """
        self._eps[str(agent_id)] = max(0.0, min(1.0, float(eps)))

    def hard_blocked(self, agent_id: str, cell: Coord) -> bool:
        """
        HARD enforcement path:
          - If True, A* will treat `cell` as an unwalkable obstacle for this agent.
          - This norm uses only SOFT enforcement, so we never hard-block.
        """
        return False

    def soft_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        """
        SOFT enforcement path:
          - Returning >0 increases the path cost for moving from `cur` to `nxt`.
          - We only penalize when `nxt` is an apple AND eating it would breach the local K-reserve.
          - With probability ε(agent), the penalty is ignored on this step.

        Design choice:
          - Planning is easier if this is *fast* and mostly deterministic.
          - We keep the RNG usage isolated to the final ε check; everything else is pure math.
        """
        apples_now = self._current_apples()
        if nxt not in apples_now:
            return 0.0

        # Count neighbors that would remain AFTER harvesting nxt (exclude nxt itself).
        if self._neighbors_excluding(nxt, apples_now) >= self.K:
            return 0.0

        # ε-compliance: skip penalty with probability ε(agent)
        eps = self._eps.get(str(agent_id), 0.0)
        rng = self._get_agent_rng(agent_id)
        if rng.random() < eps:
            return 0.0

        return self.penalty

    # ─────────────────────────────────────────────────────────────────────────────
    # Backwards-compatibility shims (so your current pipeline keeps working)
    # ─────────────────────────────────────────────────────────────────────────────

    def update_apples(self, apples: Set[Coord]) -> None:
        """
        OLD API (runner-driven): set apples explicitly each step.
        If you call this, we use this set instead of StateView.apples().
        This keeps your existing code running while we migrate to StateView.
        """
        self._apples_override = set(apples)

    def is_blocked(self, agent_id: str, cell: Coord) -> bool:
        """Alias to hard_blocked(...) for older call sites."""
        return self.hard_blocked(agent_id, cell)

    def step_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        """Alias to soft_penalty(...) for older call sites."""
        return self.soft_penalty(agent_id, cur, nxt)

    def expected_step_penalty(self, agent_id: str, cur: Coord, nxt: Coord) -> float:
        """
        Deterministic *expected* penalty useful for planning heuristics:
          - 0 if nxt is not an apple or would not breach K.
          - (1 - ε(agent)) * penalty otherwise.
        This can be used by A* if you prefer fully-deterministic costs at plan time.
        """
        apples_now = self._current_apples()
        if nxt not in apples_now:
            return 0.0
        if self._neighbors_excluding(nxt, apples_now) >= self.K:
            return 0.0
        eps = self._eps.get(str(agent_id), 0.0)
        return (1.0 - eps) * self.penalty

    # ─────────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _current_apples(self) -> Set[Coord]:
        """
        Get the current apple set from either:
          - manual override (update_apples), or
          - the bound StateView (preferred in the new pipeline).
        """
        if self._apples_override is not None:
            return self._apples_checked(self._apples_override)  # type: ignore[attr-defined]
        # Use StateView if available
        if self._state is None:
            # Defensive: if not bound yet, behave as if there are no apples.
            return set()
        return set(self._state.apples())

    def _apples_checked(self, apples: Set[Coord]) -> Set[Coord]:
        """Tiny guard to ensure the set contains tuples of (int, int)."""
        # (You can remove this if you trust upstream types.)
        checked = set()
        for a in apples:
            r, c = a
            checked.add((int(r), int(c)))
        return checked

    def _neighbors_excluding(self, cell: Coord, apples_set: Set[Coord]) -> int:
        """
        Count apples within L2 radius of 'cell', excluding the cell itself.
        We compare squared distances to avoid sqrt (faster & exact for ints).
        """
        r0, c0 = cell
        count = 0
        for (r, c) in apples_set:
            if (r, c) == (r0, c0):
                continue
            dr = r - r0
            dc = c - c0
            if (dr * dr + dc * dc) <= self.radius_sq + 1e-9:
                count += 1
        return count

    def _get_agent_rng(self, agent_id: str) -> random.Random:
        """
        Derive a stable per-agent RNG so ε-compliance is reproducible:
          seed(agent) = base_seed XOR hash(agent_id) (truncated to 32 bits).
        """
        if agent_id in self._rng_by_agent:
            return self._rng_by_agent[agent_id]
        derived = (self._base_seed ^ (hash(agent_id) & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng = random.Random(derived)
        self._rng_by_agent[agent_id] = rng
        return rng
