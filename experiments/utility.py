from word_play.presets.observation_presets import format_discussion_phase
from environments.altar.common_utility import round_num_to_time_str
# TODO: dont import everything
from experiments.configs import *
import json
import os


def full_history_to_str(results, discussion_phase_turn_count, all_agent_names) -> str:
	history_str = ''
	for round, (discussion_phase, altar_signals, actions) in enumerate(zip(results['discussion_history'], results['altar_signal_history'], results['action_history']), start=1):
		history_str += '\n\n=================================================='
		history_str += f'\nTime: {round_num_to_time_str(round)}'
		history_str += '\n=================================================='

		history_str += '\n\nALTAR SIGNALS:'	
		for altar_idx, signal in enumerate(altar_signals):
			# history_str += f"\n{signal.altar_name}'s Message: {signal.signal_message}"
			history_str += f"\n{LIST_OF_ALTAR_NAMES[altar_idx]}'s Message: {signal}"
		
		history_str += '\n\nDISCUSSION PHASE:\n\n'
		history_str += format_discussion_phase(
							discussion_messages=discussion_phase,
							discussion_phase_turn_count=discussion_phase_turn_count,
							all_agent_names=all_agent_names,
							observing_agent_id=0)	# NOTE: we hardcode the observing agent id to 0 since this function is just used to print the history after the experiment is complete
		
		history_str += '\n\n\nACTIONS:'
		for agent_idx, action in enumerate(actions):
			history_str += f'\n{all_agent_names[agent_idx]}: {action}'
	
	return history_str.strip()


def config_dict_to_file(exp_config: dict, config_name: str):
	config_path = os.path.join(EXP_CONFIGS_DIR, f'{config_name}.json')
	# TODO: using EXP_CONFIGS_DIR might not be very nice
	with open(config_path, 'w', encoding='utf-8') as f:
		json.dump(exp_config, f, indent=4)
	return config_path
