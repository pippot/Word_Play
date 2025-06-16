import argparse
import json
from experiments.exp_setup import create_simple_env
from experiments.exp_exec import run_sim
import os
import datetime
from pprint import pprint
import time
import wandb


def run_experiments(exp_config_path: str, results_dir: str, verbosity=1):
	with open(exp_config_path, 'r') as f:
		exp_config = json.load(f)

	if verbosity >= 1:
		print('Experiment Config:')
		pprint(exp_config)

	wandb.init(project='normative_agents', config=exp_config)

	env, agent_names = create_simple_env(
		foreground_agent_configs=exp_config['foreground_agent_configs'],
		
		num_anti_altar_agents=exp_config['num_anti_altar_agents'],
		anti_altar_agent_prompt=exp_config['anti_altar_agent_prompt'],
		anti_altar_agent_model_config=exp_config['anti_altar_agent_model_config'],
		
		num_altar_loving_agents=exp_config['num_altar_loving_agents'],
		altar_loving_agent_prompt=exp_config['altar_loving_agent_prompt'],
		altar_loving_agent_model_config=exp_config['altar_loving_agent_model_config'],

		fruit_tree_types=exp_config['fruit_tree_types'],
		altar_fruit_types=exp_config['altar_fruit_types'],

		discussion_phase_turn_count=exp_config['discussion_phase_turn_count'],

		verbosity=exp_config['verbosity']
	)

	exp_start_time = time.time() 

	results = run_sim(
		env=env,
		step_count=exp_config['sim_step_count']
	)

	exp_duration_in_secs = time.time() - exp_start_time
	completion_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
	
	complete_exp_info = {
		'exp_config': exp_config,
		'completion_time': completion_time,
		'exp_duration_in_secs': exp_duration_in_secs,
		'agent_names': agent_names,
		'results': results
	}

	wandb.log(complete_exp_info)

	with open(os.path.join(results_dir, f'{exp_config["experiment_name"]}_{completion_time}.json'), 'w', encoding='utf-8') as f:
		json.dump(complete_exp_info, f, indent=4, default=str)


if __name__ == '__main__':
	
	#'''
	parser = argparse.ArgumentParser(description='Settings')
	# TODO: delete hard coded paths
	parser.add_argument('--exp_config_path')
	parser.add_argument('--results_dir', default = '/h/andrei/normative_agents/results')

	args = parser.parse_args()
	
	exp_config_path = args.exp_config_path
	results_dir = args.results_dir
	#'''

	# #exp_config_path = '/h/andrei/normative_agents/experiments/exp_configs/test_config.json'
	# exp_config_path = '/h/andrei/normative_agents/experiments/exp_configs/test_config_with_human.json'
	# results_dir = '/h/andrei/normative_agents/results'

	run_experiments(exp_config_path, results_dir)
