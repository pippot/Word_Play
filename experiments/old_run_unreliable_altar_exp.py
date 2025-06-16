from experiments.configs import *
from experiments.utility import config_dict_to_file
import datetime
import os


if __name__ == '__main__':
	experiment_name = 'unreliable_altar'
	common_model_config = MED_TOP_P_LLAMA3_CHAT_70B_CONFIG

	base_exp_config = {
		"experiment_name": experiment_name,

		"foreground_agent_configs": [
			{
				"agent_type": "no_belief_memory",
				"model_config": common_model_config,
				"discussion_prompt": NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0
			}
		],
			
		"anti_altar_agent_prompt": ANTI_ALTAR_DISCUSSION_PROMPT_1_0,
		"anti_altar_agent_model_config": common_model_config,
		
		"altar_loving_agent_prompt": ALTAR_LOVING_DISCUSSION_PROMPT_1_0,
		"altar_loving_agent_model_config": common_model_config,

		"sim_step_count": 4,

		"verbosity": 2
	}

	altar_and_trees = [
		{
			"fruit_tree_types": ["apple", "banana"],
			"altar_fruit_types": ["apple", "banana"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach"],
			"altar_fruit_types": ["apple", "banana", "peach"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange"],
			"altar_fruit_types": ["apple", "banana", "peach", "orange"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange", "plum"],
			"altar_fruit_types": ["apple", "banana", "peach", "orange", "plum"]
		},
	]

	num_altar_loving_agents = [0, 1, 3, 5]
	num_anti_altar_agents = [1, 3, 5]
	discussion_phase_turn_counts = [1, 3, 5]

	
	for num_altar_loving_agent in num_altar_loving_agents:
		for altar_and_tree in altar_and_trees:
			for discussion_phase_turn_count in discussion_phase_turn_counts:
				for num_anti_altar_agent in num_anti_altar_agents:
					exp_config = base_exp_config.copy()
					exp_config.update(altar_and_tree)
					exp_config['num_altar_loving_agents'] = num_altar_loving_agent
					exp_config['num_anti_altar_agents'] = num_altar_loving_agent
					exp_config['discussion_phase_turn_count'] = discussion_phase_turn_count

					completion_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
					config_path = config_dict_to_file(exp_config, f'{experiment_name}_{completion_time}')
					
					os.system(f'sbatch run_exp_sbatch.sh {config_path}')
