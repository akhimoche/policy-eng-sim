"""
operator_funcs.py
=================

Pure, stateless helpers for spatial reasoning and one-step action planning.

How this file fits in:
- Agents (e.g., SelfishAgent) can call A* to chase targets.
- The runner converts a desired "next cell" into action tokens → integer IDs.
- This module does **not** touch the environment directly—no side effects.

What’s new (safe & optional):
- `a_star_with_norms(...)` adds two *optional* hooks:
    * hard_blocked(agent_id, cell) -> bool      # treat cell like a wall
    * soft_penalty(agent_id, cur, nxt) -> float # extra step cost (>=0)
  If you don’t pass them, behavior is identical to vanilla A*.

Key design choices:
- Keep A* generic (no imports from any “norms” code).
- Manhattan distance heuristic (4-connected grid).
- Readability > micro-optimizations.
"""

from __future__ import annotations

import heapq
from typing import Dict, Tuple, Set, List, Optional, Callable

import numpy as np

# Type alias for grid coordinates
Coord = Tuple[int, int]

# ──────────────────────────────────────────────────────────────────────────────
# Vanilla A* (unchanged behavior)
# ──────────────────────────────────────────────────────────────────────────────
def a_star(start: Coord, goal: Coord, obstacles: Set[Coord], grid_size: Tuple[int, int]) -> List[Coord]:
    """
    Classic A* from start → goal, avoiding `obstacles`.

    Returns:
        [start, ..., goal] if a path exists, else [].

    Notes:
        - 4-connected moves only (no diagonals).
        - step cost = 1 per move.
        - admissible heuristic = Manhattan distance.
    """
    if start == goal:
        return [start]

    def heuristic(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rows, cols = grid_size
    # Up, Right, Down, Left in (row, col)
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    open_set: List[Tuple[float, Coord]] = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nb = (current[0] + dx, current[1] + dy)

            # Skip out of bounds / physics obstacles
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            if nb in obstacles:
                continue

            tentative = g_score[current] + 1.0  # base step cost

            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + heuristic(nb, goal)
                heapq.heappush(open_set, (f, nb))

    return []  # no path


# ──────────────────────────────────────────────────────────────────────────────
# A* with optional "norm" hooks (non-breaking; core stays generic)
# ──────────────────────────────────────────────────────────────────────────────

# Hook type aliases for clarity
NormsBlocked = Callable[[str, Coord], bool]           # (agent_id, cell) -> bool
NormsPenalty = Callable[[str, Coord, Coord], float]   # (agent_id, cur, nxt) -> penalty >= 0


def a_star_with_norms(
    start: Coord,
    goal: Coord,
    obstacles: Set[Coord],
    grid_size: Tuple[int, int],
    *,
    agent_id: str = "agent",
    norms_active: bool = True,                       # global on/off for easy ablations
    norms_blocked: Optional[NormsBlocked] = None,    # hard: extra unwalkable cells
    norms_penalty: Optional[NormsPenalty] = None,    # soft: extra cost on cur->nxt
    max_expansions: int = 20_000,                    # safety valve
) -> List[Coord]:
    """
    A* variant that consults **optional** norm hooks.

    Behavior if no hooks / inactive:
        - If norms_active is False OR both hooks are None → identical to vanilla A*.

    Returns:
        [start, ..., goal] if reachable; [] otherwise.
    """
    # Fast path: nothing to do → defer to vanilla A*
    if (not norms_active) and norms_blocked is None and norms_penalty is None:
        return a_star(start, goal, obstacles, grid_size)

    if start == goal:
        return [start]

    def heuristic(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rows, cols = grid_size
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Local helper: unified "blocked?" check
    def is_hard_blocked(cell: Coord) -> bool:
        r, c = cell
        if not (0 <= r < rows and 0 <= c < cols):
            return True
        if cell in obstacles:
            return True
        if norms_active and norms_blocked is not None and norms_blocked(agent_id, cell):
            return True
        return False

    # Priority queue nodes: (f_score, g_score, cell)
    open_set: List[Tuple[float, float, Coord]] = []
    heapq.heappush(open_set, (heuristic(start, goal), 0.0, start))

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}

    expansions = 0
    while open_set:
        _, g_cur, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        expansions += 1
        if expansions > max_expansions:
            return []  # give up safely

        for dx, dy in neighbors:
            nb = (current[0] + dx, current[1] + dy)

            if is_hard_blocked(nb):
                continue

            step_cost = 1.0
            if norms_active and norms_penalty is not None:
                extra = float(norms_penalty(agent_id, current, nb))
                if extra < 0.0:
                    extra = 0.0  # defensive clamp
                step_cost += extra

            cand_g = g_cur + step_cost

            if cand_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = cand_g
                f = cand_g + heuristic(nb, goal)
                heapq.heappush(open_set, (f, cand_g, nb))

    return []  # frontier exhausted without reaching goal


# ──────────────────────────────────────────────────────────────────────────────
# Orientation helper
# ──────────────────────────────────────────────────────────────────────────────
def get_north(orientation: str):
    """
    Rotate the agent one step toward 'north'.

    Returns:
        (action_token, new_orientation)

    Notes:
        - This is incremental; you may need multiple calls across timesteps
          to fully rotate depending on current orientation.
    """
    if orientation == "north":
        return "NOOP", "north"
    if orientation == "east":
        return "TURN_LEFT", "north"
    if orientation == "south":
        return "TURN_LEFT", "east"
    if orientation == "west":
        return "TURN_RIGHT", "north"
    return "NOOP", "north"


# ──────────────────────────────────────────────────────────────────────────────
# Convert a planned step to an action token
# ──────────────────────────────────────────────────────────────────────────────
def move_agent(coord_init: Coord, coord_final: Coord) -> str:
    """
    Given two adjacent grid cells, return the movement token to go init → final.

    Tokens:
        "FORWARD" (row-1), "BACKWARD" (row+1), "STEP_LEFT" (col-1),
        "STEP_RIGHT" (col+1), or "NOOP" if not a valid 4-connected step.
    """
    r0, c0 = coord_init
    r1, c1 = coord_final

    if (r0, c0) == (r1, c1):
        return "NOOP"

    # Horizontal
    if r0 == r1 and c0 > c1:
        return "STEP_LEFT"
    if r0 == r1 and c0 < c1:
        return "STEP_RIGHT"

    # Vertical
    if c0 == c1 and r0 > r1:
        return "FORWARD"    # up (north)
    if c0 == c1 and r0 < r1:
        return "BACKWARD"   # down (south)

    return "NOOP"  # not a valid 4-connected step


# ──────────────────────────────────────────────────────────────────────────────
# Multi-agent one-step planner (vanilla A* for now)
# ──────────────────────────────────────────────────────────────────────────────
def get_movement_actions(
    operator_output: Dict[str, Tuple[Coord, Coord, Set[Coord]]],
    colour_dict: List[str],
    num_players: int,
    grid_size: Tuple[int, int],
    action_map: Dict[str, int],
) -> np.ndarray:
    """
    Compute **one-step actions** for each player based on planned paths.

    Inputs:
        operator_output[color] = (init_pos, goal_pos, obstacles_set)
        colour_dict[i]          = color string for player i (keys into operator_output)
        num_players             = total agents
        grid_size               = (rows, cols)
        action_map              = token -> integer ID

    Behavior:
        - Uses vanilla A* here (no norms yet) to keep current pipeline unchanged.
        - If you want norm-aware planning, swap to `a_star_with_norms` at a single callsite.

    Output:
        np.ndarray shape (num_players,), dtype=int
    """
    moves = np.zeros((num_players,), dtype=int)
    noop_id = action_map.get("NOOP", 0)

    for i in range(num_players):
        color = colour_dict[i]

        if color not in operator_output:
            moves[i] = noop_id
            continue

        init, goal, obstacles = operator_output[color]
        obstacles = set(obstacles) if obstacles is not None else set()

        path = a_star(init, goal, obstacles, grid_size)  # ← vanilla A* for now

        if not path or len(path) < 2:
            moves[i] = noop_id
            continue

        step_token = move_agent(path[0], path[1])
        moves[i] = action_map.get(step_token, noop_id)

    return moves
