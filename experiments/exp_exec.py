from environments.altar.unreliable_altar.env import Unreliable_Altar_Env
from tqdm import tqdm


def run_sim(env: Unreliable_Altar_Env, step_count: int):
	discussion_history = []
	altar_signal_history = []
	action_history = []
	reward_history = []
	info_history = []

	for step in tqdm(range(step_count), desc='steps'):
		cur_round_actions = []
		cur_round_rewards = []
		cur_round_infos = []
		
		# Discussion phase
		env.start_new_discussion_phase()
		for discussion_turn_idx in range(env.discussion_phase_turn_count):
			for agent_id, agent in enumerate(env.agents):
				observation = env.observe(agent_id)
				message, info = agent.get_discussion_message(observation)
				env.submit_message(agent_id, message)
			env.end_discussion_phase_turn()
		
		# env.end_discussion_phase() does NOT reset the cur_discussion_messages
  		# It is only reset by env.start_new_discussion_phase()
		env.end_discussion_phase()

		# Action phase
		for agent_id, agent in enumerate(env.agents):
			observation = env.observe(agent_id)
			action, info = agent.select_action(observation)
			cur_round_actions.append(action)
			cur_round_infos.append(info)
		
		env.step(cur_round_actions)

		# after performing the actions, we can check the resulting reward
		for agent_id, agent in enumerate(env.agents):
			observation, last_reward, termination, truncation, info = env.last(agent_id)
			cur_round_rewards.append(last_reward)

		discussion_history.append(env.cur_discussion_messages)
		altar_signal_history.append([elm.signal_message for elm in env.altar_signals])
		action_history.append([str(action) for action in cur_round_actions])
		reward_history.append(cur_round_rewards)
		info_history.append(cur_round_infos)

	results_dict = {
		'discussion_history': discussion_history,
		'altar_signal_history': altar_signal_history,
		'action_history': action_history,
		'reward_history': reward_history,
		'info_history': info_history,
	}

	return results_dict
