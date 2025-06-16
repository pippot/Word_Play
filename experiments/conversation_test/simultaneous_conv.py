from word_play.environment import Entity_State, Entity_Properties, Environment, Environment_State, Environment_Properties
from word_play.presets.movement_system_presets import Single_Point_Position, SINGLE_POINT_MOVEMENT_SYSTEM
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.model_presets import Human, ChatGPT
from environments.conversation_test.simultaneous_conv_env import Conversation_Test_Env
from environments.conversation_test.agents import Explicit_Belief_Conversation_Agent


def run_sim(env: Conversation_Test_Env, step_count: int):
	for step in range(step_count):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
		print('step:', step)
		this_rounds_actions = []
		this_rounds_conversation = []
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
			print('info["CoT_prompt"]:', info["CoT_prompt"])
			print('**************************************************')
			this_rounds_actions.append(action)
			this_rounds_conversation.append(info['speech_message'])
		env.step(this_rounds_actions, this_rounds_conversation)


if __name__ == '__main__':

	environment_description = "This game is all about chatting and having fun!"

	alice = Explicit_Belief_Conversation_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='Alice'
		),
		model=ChatGPT(model_name='gpt-3.5-turbo', system_prompt='You are a friendly agent.', verbosity=2),
		env_description=environment_description
	)
	bob = Explicit_Belief_Conversation_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='Bob'
		),
		#model=Human(),
		model=ChatGPT(model_name='gpt-3.5-turbo', system_prompt='You are a friendly agent.', verbosity=2),
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

	run_sim(env=env, step_count=3)
