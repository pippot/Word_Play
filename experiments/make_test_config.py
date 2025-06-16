from experiments.configs import *
from experiments.utility import config_dict_to_file


if __name__ == '__main__':
	exp_config = {
		"experiment_name": "test_experiment",

		"foreground_agent_configs": [
			{
				"agent_type": "no_belief_memory",
				"model_config": MED_TOP_P_LLAMA3_CHAT_8B_CONFIG,
				"discussion_prompt": NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0
			}
		],
			
		"num_anti_altar_agents": 1,
		"anti_altar_agent_prompt": ANTI_ALTAR_DISCUSSION_PROMPT_1_0,
		"anti_altar_agent_model_config": MED_TOP_P_LLAMA3_CHAT_8B_CONFIG,
		
		"num_altar_loving_agents": 1,
		"altar_loving_agent_prompt": ALTAR_LOVING_DISCUSSION_PROMPT_1_0,
		"altar_loving_agent_model_config": MED_TOP_P_LLAMA3_CHAT_8B_CONFIG,

		"fruit_tree_types": ["apple", "banana"],
		"alter_fruit_types": ["apple"],

		"discussion_phase_turn_count": 2,
		"sim_step_count": 2,

		"verbosity": 2
	}

	config_dict_to_file(exp_config, "test_config")
