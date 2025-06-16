from word_play.environment import Entity_State, Entity_Properties, Environment_State, Environment_Properties
from word_play.presets.movement_system_presets import Single_Point_Position, SINGLE_POINT_MOVEMENT_SYSTEM
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.model_presets import Human, ChatGPT
from environments.altar.unreliable_altar.env import Unreliable_Altar_Env
from environments.altar.unreliable_altar.agents import Anti_Altar_Agent, get_anti_altar_agent_system_prompt
from environments.altar.common_entities import Altar, Altar_Properties, Altar_Signal_Type, Fruit_Tree, Fruit_Tree_Properties


def run_sim(env: Unreliable_Altar_Env, step_count: int):
	for step in range(step_count):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
		print('step:', step)
		this_rounds_actions = []
		
		# Discussion phase
		env.start_new_discussion_phase()
		for discussion_turn_idx in range(env.discussion_phase_turn_count):
			for agent_id, agent in enumerate(env.agents):
				print('||||||||||||||||||||||||||||||||||||||||||||||||||')
				print('step:', step)
				print('discussion_turn_idx:', discussion_turn_idx)
				print('agent_id:', agent_id)
				print('||||||||||||||||||||||||||||||||||||||||||||||||||')
				observation = env.observe(agent_id)
				message, info = agent.get_discussion_message(observation)
				print('--------------------------------------------------')
				print('--------------------------------------------------')
				# print('observation:', observation)
				print('info["discussion_message_prompt"]:', info["discussion_message_prompt"])
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

	alice = Anti_Altar_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='Alice'
		),
		#discussion_model=Human(),
		discussion_model=ChatGPT(
			model_name='gpt-4-turbo',
			system_prompt=get_anti_altar_agent_system_prompt('Alice'),
			verbosity=2),
	)
	bob = Anti_Altar_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='Bob'
		),
		# discussion_model=Human(),
		discussion_model=ChatGPT(
			model_name='gpt-4-turbo',
			system_prompt=get_anti_altar_agent_system_prompt('Bob'),
			verbosity=2),
	)

	altar = Altar(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Altar_Properties(
			name='Chieftain Orion',
			signal=Altar_Signal_Type.APPLE
		)
	)

	apple_tree = Fruit_Tree(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Fruit_Tree_Properties(
			name='apple tree',
			fruit='apple'
		)
	)
	banana_tree = Fruit_Tree(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Fruit_Tree_Properties(
			name='banana tree',
			fruit='banana'
		)
	)


	env = Unreliable_Altar_Env(
		state=Environment_State(
			entities=[alice, bob, altar, apple_tree, banana_tree]
		),
		properties=Environment_Properties(
			description='In this environment all community members are part of a clan lead by chief.'
		),
		movement_system=SINGLE_POINT_MOVEMENT_SYSTEM,
		reward_func=zero_reward_func
	)

	run_sim(env=env, step_count=2)
