from word_play.environment import Entity_State, Entity_Properties, Environment, Environment_State, Environment_Properties
from word_play.presets.movement_system_presets import Single_Point_Position, SINGLE_POINT_MOVEMENT_SYSTEM
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.model_presets import Human, ChatGPT
from environments.conversation_test.discussion_phase_env import Conversation_Test_Env 
from environments.conversation_test.agents import Explicit_Belief_Discussion_Phase_Agent


def run_sim(env: Conversation_Test_Env, step_count: int):
	for step in range(step_count):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
		print('step:', step)
		this_rounds_actions = []
		
		# Discussion phase
		env.start_new_discussion_phase()
		for discussion_turn_idx in range(env.discussion_phase_turn_count):
			for agent_id, agent in enumerate(env.agents):
				observation = env.observe(agent_id)
				message, info = agent.get_discussion_message(observation)
				#print('--------------------------------------------------')
				#print('--------------------------------------------------')
				#print('observation:', observation)
				#print('info["discussion_message_prompt"]:', info["discussion_message_prompt"])
				env.submit_message(agent_id, message)
			env.end_discussion_phase_turn()
		env.end_discussion_phase()

		# Action phase
		for agent_id, agent in enumerate(env.agents):
			observation = env.observe(agent_id)
			action, info = agent.select_action(observation)
			# print('observation.conversation:', observation.conversation)
			# print('observation.observing_agent_id:', observation.observing_agent_id)
			# print('observation:', observation)
			# print('')
			# print('agent_id:', agent_id)
			# print('action:', action)
			# print('info["speech_message"]:', info["speech_message"])
			#print('info["CoT_prompt"]:', info["CoT_prompt"])
			#print('**************************************************')
			this_rounds_actions.append(action)
		
		env.step(this_rounds_actions)


if __name__ == '__main__':

	environment_description = "This game is all about chatting and having fun! The cycles between discussion and action phases.\
 Each disucssion phase is split up into 3 turns during which everyone has a chance to say something. Make sure reply to messages\
 from previous turns!"

	alice = Explicit_Belief_Discussion_Phase_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='Alice'
		),
		model=Human(),
		#model=ChatGPT(model_name='gpt-3.5-turbo', system_prompt='You are Alice.', verbosity=2),
		env_description=environment_description
	)
	bob = Explicit_Belief_Discussion_Phase_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='Bob'
		),
		model=Human(),
		# model=ChatGPT(model_name='gpt-4-turbo', system_prompt='You are Bob.', verbosity=2),
		env_description=environment_description
	)


	env = Conversation_Test_Env(
		state=Environment_State(
			entities=[alice, bob]
		),
		properties=Environment_Properties(
			description=environment_description
		),
		movement_system=SINGLE_POINT_MOVEMENT_SYSTEM,
		reward_func=zero_reward_func
	)

	run_sim(env=env, step_count=2)
