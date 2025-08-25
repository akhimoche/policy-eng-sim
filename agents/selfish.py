# --------------------------------------------------------------------
# SelfishAgent — A hard-coded heuristic agent for Melting Pot
# --------------------------------------------------------------------
# PURPOSE:
#   - Represents a single agent that tries to collect the *nearest reachable apple*
#     using A* pathfinding, avoiding obstacles.
#   - Does NOT learn; instead follows a fixed algorithm.
#   - Assumes agents have been "calibrated" to face NORTH before the main loop starts.
#
# CONNECTIONS TO OTHER FILES:
#   - Inherits from BaseAgent (agents/base_agent.py) to get:
#       * self.id          — agent identifier
#       * self.rng         — per-agent random number generator
#       * self.action_map  — mapping from action tokens to integer IDs
#   - Instantiated in scripts/run_agents.py:
#       * Given action range (a_min, a_max), the converter object (for pixel → symbol parsing),
#         and its unique color label (from calibration).
#   - Uses the converter from env/mp_llm_env.py to convert the environment’s RGB frame
#     into a symbolic "state" (dictionary of object labels → positions).
#   - Uses utils/operator_funcs.a_star (or a_star_with_norms) for pathfinding.
# --------------------------------------------------------------------

from .base_agent import BaseAgent, ACTION_MAP
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# NOTE: we import BOTH versions of A*: vanilla and the one with optional norm hooks.
from utils.operator_funcs import a_star as op_a_star, a_star_with_norms

# Import only the interface for norms; the concrete norm is injected from the runner.
from utils.norms.base import NormSpec

# Translation-only actions (move in the grid without turning or zapping).
# These correspond to:
#   1 = FORWARD, 2 = BACKWARD, 3 = STEP_LEFT, 4 = STEP_RIGHT
FALLBACK_TRANSLATIONS = np.array([1, 2, 3, 4], dtype=int)


class SelfishAgent(BaseAgent):
    """
    Heuristic agent that:
      - Finds the nearest reachable apple using A* search.
      - Takes the first step along the optimal path.
      - If no apple is reachable, chooses a random movement action.

    Norm wiring (optional):
      - When `use_norms=True`, pathfinding consults two *generic* callbacks:
          * hard_blocked(agent_id, cell) -> bool      # treat like a wall (spatial or temporal)
          * soft_penalty(agent_id, cur, nxt) -> float # add extra step cost (>=0)
        This *does not* bake any specific norm into core code; it’s inversion of control.
    """

    def __init__(
        self,
        agent_id: int,
        action_min: int,
        action_max: int,
        converter,       # Object that can turn RGB frames into symbolic states
        color: str,      # Agent's color label from calibration (e.g., "red")
        seed: int | None = None,
        # ---- optional norm wiring (defaults keep behavior identical to before) ----
        use_norms: bool = False,          # enable norm-aware planning
        reserve_K: int = 3,               # (kept for compatibility; ignored if norm is injected)
        reserve_radius: float = 2.0,      # (kept for compatibility; ignored if norm is injected)
        norm_penalty: float = 5.0,        # (kept for compatibility; ignored if norm is injected)
        epsilon: float = 0.0,             # ε-compliance: prob. to ignore soft penalty
        norm: Optional[NormSpec] = None   # injected norm implementation (preferred)
    ):
        # Base class sets RNG and action map
        super().__init__(agent_id, seed=seed, action_map=ACTION_MAP)

        # Store agent-specific info
        self.agent_id = agent_id
        self.action_min = int(action_min)
        self.action_max = int(action_max)
        self.converter = converter
        self.color = color

        # --- Optional norm (injected) ---
        self.use_norms = bool(use_norms)
        self.norm: Optional[NormSpec] = norm

        # If norms are enabled, ensure we have one and set ε if the norm supports it
        if self.use_norms:
            if self.norm is None:
                # You chose norm-aware planning but did not inject a norm from the runner.
                # We raise here to make the wiring error explicit instead of silently falling back.
                raise ValueError("SelfishAgent(use_norms=True) requires an injected `norm` implementing NormSpec.")
            if hasattr(self.norm, "set_epsilon"):
                try:
                    self.norm.set_epsilon(str(self.id), float(epsilon))  # type: ignore[attr-defined]
                except Exception:
                    # If a norm chooses not to implement epsilon, ignore quietly.
                    pass

    # ----------------------------------------------------------------
    # HELPER: Convert observation frame → symbolic state dictionary
    # ----------------------------------------------------------------
    def _symbolic_state(self, obs) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extracts the 'WORLD.RGB' frame from the observation and
        converts it into a symbolic state using the converter.

        Returns:
            state: dict mapping object labels (e.g., "apple", "wall", "p_red_north")
                   to lists of (row, col) positions on the grid.
        """
        frame = obs.observation[0]["WORLD.RGB"]
        state = self.converter.image_to_state(frame)["global"]
        return state

    # ----------------------------------------------------------------
    # HELPER: Collect positions for labels matching a condition
    # ----------------------------------------------------------------
    @staticmethod
    def _collect_positions(state: Dict[str, List[Tuple[int, int]]], key_pred) -> List[Tuple[int, int]]:
        """
        Loops over all labels in the symbolic state and collects positions
        for those whose label satisfies `key_pred` (a boolean test function).
        """
        out: List[Tuple[int, int]] = []
        for k, v in state.items():
            if key_pred(k):
                out.extend(v)
        return out

    # ----------------------------------------------------------------
    # HELPER: Calculate grid size (rows, cols) from state
    # ----------------------------------------------------------------
    @staticmethod
    def _grid_size(state: Dict[str, List[Tuple[int, int]]]) -> Tuple[int, int]:
        """
        Determines the grid dimensions from all positions in the state.
        Needed for bounds-checking in A*.
        """
        max_r = 0
        max_c = 0
        for positions in state.values():
            for r, c in positions:
                if r > max_r: max_r = r
                if c > max_c: max_c = c
        return (max_r + 1, max_c + 1)  # +1 because coordinates are 0-indexed

    # ----------------------------------------------------------------
    # MAIN: Decide what action to take this step
    # ----------------------------------------------------------------
    def act(self, obs):
        """
        Main decision loop:
          1. Parse the observation into a symbolic state.
          2. Find own position by matching color label.
          3. Identify all apple positions.
          4. Identify obstacles (walls, trees).
          5. Plan to each candidate apple; pick the minimal effective cost:
               steps + (expected final-step penalty if norms enabled)
          6. Execute the first step on the chosen path; fallback random if no path.

        NOTE on norms:
          - We update the norm’s apple set each step (old API) so it has read-only context.
          - When `use_norms=True`, we call `a_star_with_norms` and pass the two norm callbacks.
          - This keeps core planner generic while allowing spatial/temporal rules via hooks.
        """
        # Step 1: Get symbolic map of the world
        state = self._symbolic_state(obs)

        # Step 2: Locate this agent's position by its color label
        desired_prefix = f"p_{self.color}_"
        starts_any = self._collect_positions(state, lambda k: k.startswith(desired_prefix))
        start: Optional[Tuple[int, int]] = starts_any[0] if starts_any else None

        # If we can't find ourselves → move randomly
        if start is None:
            return int(self.rng.choice(FALLBACK_TRANSLATIONS))

        # Step 3: Find apples (prefer exact live label; fallback to substring)
        apples = state.get("apple", [])
        if not apples:
            # fallback, but exclude any 'wait' variants
            apples = self._collect_positions(state, lambda k: ("apple" in k) and ("wait" not in k.lower()))

        # Step 4: Build obstacle set
        walls  = set(self._collect_positions(state, lambda k: "wall" in k))
        trees  = set(self._collect_positions(state, lambda k: "tree" in k))
        obstacles: Set[Tuple[int, int]] = walls | trees

        # Step 5: Compute grid size
        grid_size = self._grid_size(state)

        # (Norm hook context) keep the norm's apple set up to date (old API; fine for now)
        # This gives the norm read-only knowledge of where apples currently are.
        if self.use_norms and self.norm is not None and hasattr(self.norm, "update_apples"):
            try:
                self.norm.update_apples(set(apples))  # type: ignore[attr-defined]
            except Exception:
                pass

        # Step 6: Evaluate each apple by "effective cost" and keep the best path.
        best_path: Optional[List[Tuple[int, int]]] = None
        best_score = float("inf")
        eid = str(self.id)

        for goal in apples:
            # Plan a path to this apple.
            if self.use_norms and self.norm is not None:
                # Norm-aware planning: planner consults optional hooks.
                path = a_star_with_norms(
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                    grid_size=grid_size,
                    agent_id=eid,
                    norms_active=True,
                    norms_blocked=getattr(self.norm, "hard_blocked", lambda *_: False),
                    norms_penalty=getattr(self.norm, "soft_penalty", lambda *_: 0.0),
                )
            else:
                # Vanilla planning (identical to your previous behavior)
                path = op_a_star(start, goal, obstacles, grid_size)

            # Treat [] or a 1-node path as "no usable path"
            if not path or len(path) < 2:
                continue

            # Base cost = number of steps (edges)
            base_steps = len(path) - 1

            # Add expected penalty for the *final step* into the apple (deterministic) if provided
            exp_pen = 0.0
            if self.use_norms and self.norm is not None and hasattr(self.norm, "expected_step_penalty"):
                try:
                    exp_pen = float(self.norm.expected_step_penalty(eid, path[-2], path[-1]))  # type: ignore[attr-defined]
                except Exception:
                    exp_pen = 0.0

            score = base_steps + exp_pen

            # Keep the best (lowest) effective score
            if score < best_score:
                best_score = score
                best_path = path

        # Step 7: No reachable apple → fallback random move
        if not best_path or len(best_path) < 2:
            return int(self.rng.choice(FALLBACK_TRANSLATIONS))

        # Step 8: Take the first step toward the apple (map delta to action id)
        start_r, start_c = start
        next_r, next_c = best_path[1]
        dr = next_r - start_r
        dc = next_c - start_c

        if dc == -1:
            action_id = 1  # FORWARD (north)
        elif dc == 1:
            action_id = 2  # BACKWARD (south)
        elif dr == -1:
            action_id = 3  # STEP_LEFT (west)
        elif dr == 1:
            action_id = 4  # STEP_RIGHT (east)
        else:
            action_id = int(self.rng.choice(FALLBACK_TRANSLATIONS))  # safety fallback

        return action_id
