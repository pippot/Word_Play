from experiments.configs import *
from experiments.utility import config_dict_to_file
import datetime
import os
from tqdm import tqdm
import copy
import json
import pandas as pd


if __name__ == '__main__':
	results_dir = "/h/andrei/normative_agents/results"
	
	experiment_name = 'competing_altars'
	# experiment_name = 'unreliable_altar'
	# extra_exp_id = 'final_2'
	# extra_exp_id = 'rebuttal_1'
	extra_exp_id = 'normatively_prompted'
	common_model_config = MED_TOP_P_LLAMA3_CHAT_8B_CONFIG
	# common_model_config = MED_TOP_P_LLAMA3_MULTILINGUAL_8B_CONFIG
	# common_model_config = HUMAN_MODEL_CONFIG

	base_exp_config = {
		"experiment_name": experiment_name,
		"extra_exp_id": extra_exp_id,
	
		"anti_altar_agent_prompt": ANTI_ALTAR_DISCUSSION_PROMPT_1_0,
		"anti_altar_agent_model_config": common_model_config,
		
		"altar_loving_agent_prompt": ALTAR_LOVING_DISCUSSION_PROMPT_1_0,
		"altar_loving_agent_model_config": common_model_config,

		# "sim_step_count": 8,
		"sim_step_count": 4,

		"verbosity": 2
	}

	#'''
	altar_and_trees = [
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange"],
			"altar_fruit_types": ["apple", "banana", "peach", "orange"]
		},
		{
			"fruit_tree_types": ["apple", "banana"],
			"altar_fruit_types": ["apple", "banana"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange", "plum"],
			"altar_fruit_types": ["apple", "banana", "peach", "orange", "plum"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach"],
			"altar_fruit_types": ["apple", "banana", "peach"]
		},
	]
	#'''
	'''
	altar_and_trees = [
		{
			"fruit_tree_types": ["apple", "banana"],
			"altar_fruit_types": ["apple"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange"],
			"altar_fruit_types": ["apple"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach"],
			"altar_fruit_types": ["apple"]
		},
		{
			"fruit_tree_types": ["apple", "banana", "peach", "orange", "plum"],
			"altar_fruit_types": ["apple"]
		},
	]
	#'''

	foreground_agents = [
		# {
		# 	"foreground_agent_configs": [
		# 		{
		# 			"agent_type": "normative",
		# 			"model_config": common_model_config,
		# 			"discussion_prompt": NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0,
		# 		}
		# 	],
		# },
		{
			"foreground_agent_configs": [
				{
					"agent_type": "no_belief_memory",
					"model_config": common_model_config,
					"discussion_prompt": NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0,
				}
			],
		},
	]

	# discussion_phase_turn_counts = [1, 3, 5]
	# discussion_phase_turn_counts = [1, 2, 3]
	discussion_phase_turn_counts = [1]
	num_altar_loving_agents = [1, 3, 5]
	# num_altar_loving_agents = [0]
	# num_anti_altar_agents = [1, 3, 5]
	# num_anti_altar_agents = [3, 5]
	num_anti_altar_agents = [0]
	# trail_repeats = 5
	# trail_repeats = 4
	trail_repeats = 3

	kwargs = [
				{
					'normative_weight': 0.5,
				},
				# {
				# 	'normative_weight': 0.1,
				# },
				# {
				# 	'normative_weight': 0.9,
				# },
				# {
				# 	'normative_weight': 0.25,
				# },
	]




	# load df of existing results

	results_path = results_dir

	# Load and combine the JSON files into a single DataFrame
	data_frames = []
	for file_name in os.listdir(results_path):
		with open(os.path.join(results_path, file_name), 'r') as file:
			try:
				data = json.load(file)
			except json.decoder.JSONDecodeError:
				# some json result files didnt save correctly
				print('Error loading JSON file:', file_name)
				continue
			cur_df = pd.json_normalize(data)
			data_frames.append(cur_df)

	# Concatenate all data frames
	full_df = pd.concat(data_frames, ignore_index=True)
	
	full_df['num_altars'] = full_df['exp_config.altar_fruit_types'].apply(len)
	full_df['num_fruit_trees'] = full_df['exp_config.fruit_tree_types'].apply(len)

	df = full_df[
		(full_df['exp_config.experiment_name'] == experiment_name) &
		(full_df['exp_config.extra_exp_id'] == extra_exp_id) #&
		# (full_df['exp_config.extra_exp_id'] == "unreliable_altar_feasibility_5") #&
		
		# (full_df['exp_config.experiment_name'] == "competing_altars") &
		# (full_df['exp_config.extra_exp_id'] == "final_2") #&
		# (full_df['exp_config.extra_exp_id'] == "final") #&
		
		# (full_df['exp_config.extra_exp_id'] == "testing_normative_feasibility_3") #&
		# (full_df['exp_config.discussion_phase_turn_count'] == 1) &
		# TODO: this isn't doing anything atm because it is saved in the wrong place
		# (full_df['exp_config.kwargs.normative_weight'] == 0.5)
	]

	# Function to extract the keys and create new columns
	def expand_config_columns(row):
		first_config = row['exp_config.foreground_agent_configs'][0]
		for key, value in first_config.items():
			row[f'exp_config.foreground_agent_first_config.{key}'] = value
		return row

	if not df.empty:
		# Apply the function to each row
		df = df.apply(expand_config_columns, axis=1)

		# Drop the original column if no longer needed
		df.drop(columns=['exp_config.foreground_agent_configs'], inplace=True)



	# for repeat_idx in range(trail_repeats):
	for kwarg in kwargs:
		for discussion_phase_turn_count in discussion_phase_turn_counts:
			for num_altar_loving_agent in num_altar_loving_agents:
				for num_anti_altar_agent in num_anti_altar_agents:
					for altar_and_tree in altar_and_trees:
						for agent_config in foreground_agents:
							for repeat_idx in range(trail_repeats):

								# check if experiment already exists
								if not df.empty:
									condition = (df['exp_config.num_altar_loving_agents'] == num_altar_loving_agent) & \
												(df['exp_config.num_anti_altar_agents'] == num_anti_altar_agent) & \
												(df['exp_config.discussion_phase_turn_count'] == discussion_phase_turn_count) & \
												(df['num_fruit_trees'] == len(altar_and_tree['fruit_tree_types'])) & \
												(df['num_altars'] == len(altar_and_tree['altar_fruit_types'])) & \
												(df['exp_config.foreground_agent_first_config.agent_type'] == agent_config['foreground_agent_configs'][0]['agent_type'])
								
									exitings_exp_count = df.loc[condition].shape[0]

								else:
									exitings_exp_count = 0

								if exitings_exp_count >= trail_repeats:
									print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
									print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
									print('SKIPPED')
									print('exitings_exp_count:', exitings_exp_count)
									print(df.loc[condition])
									continue
								else:
									print('RUNNING')
									print('exitings_exp_count:', exitings_exp_count)
									# continue



								exp_config = copy.deepcopy(base_exp_config)
								cur_agent_config = copy.deepcopy(agent_config)
								
								for agent_idx in range(len(cur_agent_config['foreground_agent_configs'])):
									if 'kwargs' in cur_agent_config['foreground_agent_configs'][agent_idx]:
										cur_agent_config['foreground_agent_configs'][agent_idx]['kwargs'].update(kwarg)
									else:
										cur_agent_config['foreground_agent_configs'][agent_idx]['kwargs'] = kwarg
								
								exp_config.update(altar_and_tree)
								exp_config.update(cur_agent_config)

								exp_config['num_altar_loving_agents'] = num_altar_loving_agent
								exp_config['num_anti_altar_agents'] = num_anti_altar_agent
								exp_config['discussion_phase_turn_count'] = discussion_phase_turn_count
								exp_config['repeat_idx'] = repeat_idx

								completion_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
								config_path = config_dict_to_file(exp_config, f'{experiment_name}_{completion_time}')
								
								# os.system(f'sbatch run_exp_sbatch.sh {config_path}')
								os.system(f'python run_exp.py --exp_config_path={config_path} --results_dir={results_dir}')
