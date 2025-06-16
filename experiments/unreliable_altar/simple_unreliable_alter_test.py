from experiments.exp_setup import create_simple_env
from experiments.exp_exec import run_sim
from experiments.utility import full_history_to_str
# TODO: dont import everything
from experiments.configs import *


if __name__ == '__main__':
	# all these values should likely be moved to a config file

	common_model_config = MED_TOP_P_LLAMA3_CHAT_8B_CONFIG

	foreground_agent_config = {
		'agent_type': 'normative',
		'model_config': common_model_config,
		'discussion_prompt': NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0
	}

	num_anti_altar_agents = 1
	anti_altar_agent_model_config = common_model_config
	anti_altar_agent_prompt = ANTI_ALTAR_DISCUSSION_PROMPT_1_0

	num_altar_loving_agents = 0
	altar_loving_agent_model_config = common_model_config
	altar_loving_agent_prompt = ALTAR_LOVING_DISCUSSION_PROMPT_1_0

	fruit_tree_types = ['apple', 'banana']
	altar_fruit_types = ['apple']

	discussion_phase_turn_count = 2
	sim_step_count = 2
	verbosity = 2


	env, agent_names = create_simple_env(
		foreground_agent_configs=[foreground_agent_config],
		
		num_anti_altar_agents=num_anti_altar_agents,
		anti_altar_agent_prompt=anti_altar_agent_prompt,
		anti_altar_agent_model_config=anti_altar_agent_model_config,
		
		num_altar_loving_agents=num_altar_loving_agents,
		altar_loving_agent_prompt=altar_loving_agent_prompt,
		altar_loving_agent_model_config=altar_loving_agent_model_config,

		fruit_tree_types=fruit_tree_types,
		altar_fruit_types=altar_fruit_types,

		discussion_phase_turn_count=discussion_phase_turn_count,

		verbosity=verbosity
	)

	results = run_sim(env=env, step_count=sim_step_count)


	print(full_history_to_str(
		results=results,
		discussion_phase_turn_count=discussion_phase_turn_count,
		all_agent_names=agent_names
	))
