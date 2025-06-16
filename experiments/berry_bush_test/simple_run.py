from word_play.environment import Entity_State, Entity_Properties, Environment, Environment_State, Environment_Properties
from word_play.presets.movement_system_presets import Position_2D, INFINITE_2D_MOVEMENT_SYSTEM
from word_play.presets.reward_func_presets import zero_reward_func
from environments.berry_bust_test.env import Berry_Patch_Env
from environments.berry_bust_test.entities import BerryBush, BerryBushProperties
from environments.berry_bust_test.agents import Random_Berry_Agent


def run_sim(env: Environment, step_count: int):
	for step in range(step_count):
		# print('step:', step)
		this_rounds_actions = []
		for agent_id, agent in enumerate(env.agents):
			observation = env.observe(agent_id)
			action, info = agent.select_action(observation)
			print('>>>>>>>>>>>>>>> info start >>>>>>>>>>>>>>>')
			print('agent_id:', agent_id)
			print('observation:', observation)
			print('action:', action)
			print('<<<<<<<<<<<<<<< info end <<<<<<<<<<<<<<<')
			this_rounds_actions.append(action)
		env.step(this_rounds_actions)


if __name__ == '__main__':
	berry_bush = BerryBush(
		state=Entity_State(
			position=Position_2D(x=1, y=2)
		),
		properties=BerryBushProperties(
			name='berry bush',
			berry_type='crunchy red'
		)
	)
	print('berry_bush.properties:', berry_bush.properties)

	print('berry_bush.get_all_exposed_action_descriptions():', berry_bush.get_all_exposed_action_descriptions())

	random_agent = Random_Berry_Agent(
		state=Entity_State(
			position=Position_2D(x=1, y=2)
		),
		properties=Entity_Properties(
			name='random agent'
		)
	)

	env = Berry_Patch_Env(
		state=Environment_State(
			entities=[berry_bush, random_agent]
		),
		properties=Environment_Properties(
			description="The Berry Patch is where magic happens!"
		),
		movement_system=INFINITE_2D_MOVEMENT_SYSTEM,
		reward_func=zero_reward_func
	)
	
	run_sim(env=env, step_count=3)
