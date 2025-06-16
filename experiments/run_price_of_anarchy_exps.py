from experiments.configs import *
from experiments.utility import config_dict_to_file
import datetime
import os


if __name__ == '__main__':
	experiment_name = 'price_of_anarchy'
	common_model_config = MED_TOP_P_LLAMA3_CHAT_70B_CONFIG

	base_exp_config = {
		"experiment_name": experiment_name,

		"num_anti_altar_agents": 0,
		"anti_altar_agent_prompt": ANTI_ALTAR_DISCUSSION_PROMPT_1_0,
		"anti_altar_agent_model_config": common_model_config,
		
		"num_altar_loving_agents": 0,
		"altar_loving_agent_prompt": ALTAR_LOVING_DISCUSSION_PROMPT_1_0,
		"altar_loving_agent_model_config": common_model_config,
	
		"sim_step_count": 4,

		"verbosity": 2
	}

	foreground_agent = {
		"agent_type": "no_belief_memory",
		"model_config": common_model_config,
		"discussion_prompt": NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0
	}
	
	tree_options = [
		{
			"fruit_tree_types": ["apple", "banana"],
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach"],
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange"],
		},
	]

	altar_fruits = ["apple"]

	num_foreground_agents = [2, 3, 4]
	discussion_phase_turn_counts = [1, 3, 5]

	
	for trees in tree_options:
		for discussion_phase_turn_count in discussion_phase_turn_counts:
			for foreground_agent_count in num_foreground_agents:
				for altar_present in [True, False]:
					exp_config = base_exp_config.copy()

					exp_config.update(trees)
					exp_config['discussion_phase_turn_count'] = discussion_phase_turn_count
					# NOTE: this is creating a list with foreground_agent_count references to the same dict
					exp_config['foreground_agent_configs'] = [foreground_agent] * foreground_agent_count
					if altar_present:
						exp_config['altar_fruit_types'] = altar_fruits
					else:
						exp_config['fruit_tree_types'] = []

					completion_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
					config_path = config_dict_to_file(exp_config, f'{experiment_name}_{completion_time}')
					
					os.system(f'sbatch run_exp_sbatch.sh {config_path}')
