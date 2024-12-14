from meltingpot import substrate
from matplotlib import pyplot as plt
from env.mp_llm_env import LLMPrepObject
import numpy as np
import utils.proc_funcs


# Define the substrate (environment) name
env_name = 'coins'  # example environment from Melting Pot
num_players=2
roles = tuple(['default' for _ in range(num_players)])
# Build the environment using the substrate's configuration
env = substrate.build(env_name, roles=roles)
converter = LLMPrepObject('/home/akhimoche/meta-ssd/sprite_labels/coins')
timestep_dictionary = {"Timestep 0": {}}

# Retrieve the action_spec from the environment
action_spec = env.action_spec()[0]
num_actions = action_spec.num_values
action_max = action_spec.maximum
action_min = action_spec.minimum
print(action_max)
print(action_min)

obs = env.reset()
done = False
t=0
window_size = 10
while not done:
    # Example: random actions for each agent
    #actions = np.random.randint(action_min,action_max+1, num_players)
    actions = np.ones((num_players), dtype =int)
    # Step through the environment
    timestep = env.step(actions)

    # Get world RGB from 0th agent (assuming full observability)
    screen_frame = timestep.observation[0]['WORLD.RGB']

    # Get dictionary of states from world RGB frame
    processed = converter.image_to_state(screen_frame)['global']
    print(utils.proc_funcs.get_dynamic_info(processed))
    plt.imshow(screen_frame)
    plt.show()
    # total system reward is social welfare
    processed["social_welfare"] = [sum(timestep.reward).item()] # turn into list for iteration in get_dynamic info

    #print(utils.proc_funcs.get_dynamic_info(processed))
    timestep_dictionary[f"Timestep {t+1}"] = processed

    # Check if episode is done
    done = timestep.last()
    t+=1

    if t==window_size:

        break


result = utils.proc_funcs.dynamic_process(timestep_dictionary)
print(f"\n Dynamic Process after {window_size} steps \n")

for i in range(len(result)):
     print(f"t{i+1}:", end="")
     print(result[f"t_{i+1}"])
