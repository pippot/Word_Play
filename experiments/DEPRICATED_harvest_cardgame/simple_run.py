from word_play.environments_OLD.harvest_cardgame.agents import Constant_Strategy_Agent, Explicit_Belief_Agent
from word_play.environments_OLD.harvest_cardgame.env import Harvest_Action
from word_play.environments_OLD.harvest_cardgame.env import ActOpt, Harvest
from word_play.model import Human, ChatGPT
from tqdm import tqdm


# TODO: split this up into some nice functions
def run_sim(env, step_count):

	env.reset(seed=42)

	action_history = []
	reward_history = []
	expectation_history = []
	cot_history = []

	for step in tqdm(range(step_count), desc='steps'):
		actions = []
		action_history.append([None, None])
		expectation_history.append([None, None])
		cot_history.append([None, None])
		# reward is offset by 1 step
		if step > 0: reward_history.append([None, None])

		for agent_id, agent in tqdm(enumerate(env.agent_iter()), total=env.agent_count, desc='agents'):
			observation, reward, termination, info = env.observe(agent_id)
			
			# reward is offset by 1 step
			if step > 0: reward_history[-1][agent_id] = reward

			if termination:
				actions.append(None)
			else:
				for attempt in range(5):
					# try:
						action, extra_info = agent.select_action(observation, reward)
						break
					# except Exception as e:
					# 	# gpt output an invalid format
					# 	print('invalid format on attempt:', attempt)
					# 	print('exception:', e)
					# 	pass
				actions.append(action)
				action_history[-1][agent_id] = action
				expectation_history[-1][agent_id] = extra_info['new_expectations']
				cot_history[-1][agent_id] = extra_info['selection_CoT']
				
		env.step(actions)

	# adding final reward since rewards are offset by 1
	reward_history.append([None, None])
	for agent_id in range(env.agent_count):
		observation, reward, termination, info = env.observe(agent_id)
		reward_history[-1][agent_id] = reward

	return action_history, reward_history, expectation_history, cot_history


if __name__ == '__main__':
	
	agents = [
		#Constant_Strategy_Agent(Action(ActOpt.apple, ActOpt.banana)),
		Constant_Strategy_Agent(constant_action=Harvest_Action(ActOpt.banana, ActOpt.apple)),
		Explicit_Belief_Agent(Human()),
		#Explicit_Belief_Agent(ChatGPT('gpt-3.5-turbo', 'You are a strategic game playing agent.', verbosity=1)),
		#Explicit_Belief_Agent(ChatGPT('gpt-3.5-turbo', 'You are a strategic game playing agent.', verbosity=1)),
	]
	env = Harvest(agents=agents, institutional_signal="We want to increase the production bananas.")
	
	run_sim(env=env, step_count=3)