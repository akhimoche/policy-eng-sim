# scripts/compare_norms_once.py
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# plotting
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# env + utils
from meltingpot import substrate
from agents.base_agent import ACTION_MAP
from agents.selfish import SelfishAgent
from env.mp_llm_env import LLMPrepObject
from utils.proc_funcs import get_changes_diff
from utils.operator_funcs import get_north

# ----- config (tweak here) -----
ENV_NAME      = "commons_harvest__open"
NUM_PLAYERS   = 5
WINDOW_SIZE   = 100
RESERVE_K     = 3
RESERVE_RAD   = 2.0
NORM_PENALTY  = 20.0    # big enough to make differences obvious

def create_env():
    roles = tuple(["default"] * NUM_PLAYERS)
    return substrate.build(ENV_NAME, roles=roles)

def calibrate_all(env, converter):
    """Return dicts: colour_dict, orientation_dict, position_dict; rotate all to north."""
    obs = env.reset()
    colour_dict, orientation_dict, position_dict = {}, {}, {}

    # Identify each player by turning them left in isolation
    for t in range(NUM_PLAYERS):
        test_actions = np.zeros(NUM_PLAYERS, dtype=int)
        test_actions[t] = ACTION_MAP["TURN_LEFT"]

        pre  = converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
        obs  = env.step(test_actions)
        post = converter.image_to_state(obs.observation[0]['WORLD.RGB'])["global"]
        changes = get_changes_diff(pre, post)

        sprite_key = next(k for k in changes if k.startswith("p_"))
        colour, orient = sprite_key.split("_")[1:3]
        colour_dict[t]      = colour
        orientation_dict[t] = orient
        position_dict[t]    = changes[sprite_key][0]

    # Rotate everyone to face north
    north_done = False
    while not north_done:
        rotation = np.zeros(NUM_PLAYERS, dtype=int)
        for t in range(NUM_PLAYERS):
            act_token, new_orient = get_north(orientation_dict[t])
            rotation[t] = ACTION_MAP[act_token]
            orientation_dict[t] = new_orient

        if rotation.sum() == 0:
            north_done = True
        else:
            obs = env.step(rotation)

    return obs, colour_dict, orientation_dict, position_dict

def run_episode(epsilon, use_norms=True, window_size=WINDOW_SIZE):
    """Return list of per-step total rewards and cumulative rewards."""
    # converter (sprite → symbols)
    labels_dir = ROOT_DIR / "utils" / "sprite_labels" / ENV_NAME
    converter = LLMPrepObject(str(labels_dir))

    # env + calibration
    env = create_env()
    obs, colour_dict, _, _ = calibrate_all(env, converter)

    # action spec
    action_spec = env.action_spec()[0]
    a_min, a_max = int(action_spec.minimum), int(action_spec.maximum)

    # instantiate agents
    agents = []
    for i in range(NUM_PLAYERS):
        agents.append(
            SelfishAgent(
                i, a_min, a_max, converter, colour_dict[i],
                use_norms=use_norms,
                epsilon=epsilon,
                reserve_K=RESERVE_K,
                reserve_radius=RESERVE_RAD,
                norm_penalty=NORM_PENALTY,
            )
        )

    # main loop
    totals = []
    done = False
    t = 0
    while not done and t < window_size:
        actions = [agent.act(obs) for agent in agents]
        timestep = env.step(actions)
        obs = timestep

        totals.append(float(sum(timestep.reward)))
        done = timestep.last()
        t += 1

    env.close()
    cum = np.cumsum(totals)
    return totals, cum

def pad_to_match(a, b):
    """Pad the shorter array by repeating its last value so plots align by x-axis."""
    la, lb = len(a), len(b)
    if la == lb: return a, b
    if la < lb:
        pad = np.full(lb - la, a[-1] if la else 0.0)
        return np.concatenate([a, pad]), b
    else:
        pad = np.full(la - lb, b[-1] if lb else 0.0)
        return a, np.concatenate([b, pad])

def main():
    # Run two scenarios
    _, cum_eps0 = run_episode(epsilon=0.0, use_norms=True,  window_size=WINDOW_SIZE)
    _, cum_eps1 = run_episode(epsilon=1.0, use_norms=True,  window_size=WINDOW_SIZE)  # ≈ “no norm”

    # Align lengths for plotting
    c0, c1 = pad_to_match(cum_eps0, cum_eps1)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(c0, label="ε = 0 (compliant)")
    ax.plot(c1, label="ε = 1 (ignores norm)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative social welfare")
    ax.set_title("Soft local-reserve norm: ε comparison (K=3, L2=2)")
    ax.grid(True)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
