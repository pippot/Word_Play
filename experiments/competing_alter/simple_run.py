from word_play.environment import Environment_State, Environment_Properties, Entity_State, Entity_Properties, Environment
from environments.competing_alter.env import Simple_Altar_Env
from environments.competing_alter.entities import Altar, Altar_Properties, Random_Alter, Random_Altar_Properties
from environments.competing_alter.agents import Obedient_Harvest_Agent, Random_Harvest_Agent, Explicit_Belief_Harvest_Agent
from environments.competing_alter.reward_func import alter_reward_func
from word_play.presets.model_presets import Human
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.movement_system_presets import Single_Point_Position
from environments.alter_common.actions import (
	Harvest_Apples_Sanction_Nothing,
	Harvest_Apples_Sanction_Apples,
	Harvest_Apples_Sanction_Bananas,
	Harvest_Bananas_Sanction_Nothing,
	Harvest_Bananas_Sanction_Apples,
	Harvest_Bananas_Sanction_Bananas,
)


def run_sim(env: Environment, step_count: int):
	for step in range(step_count):
		print('----------------')
		print('step:', step)
		this_rounds_actions = []
		for agent_id, agent in enumerate(env.agents):
			observation = env.observe(agent_id)
			action, info = agent.select_action(observation)
			print('agent_id:', agent_id)
			print('observation:', observation)
			# print('observation.alter_signals:', observation.alter_signals)
			# print('observation.last_reward:', observation.last_reward)
			# print('action:', action)
			# print('action:', action)
			print('action:', action)
			if info:
				print('info["new_expectations"]:', info["new_expectations"])
				print('info["selection_CoT"]:', info["selection_CoT"])
			this_rounds_actions.append(action)
		env.step(this_rounds_actions)


if __name__ == '__main__':
	aggressive_banana_alter = Altar(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Altar_Properties(
			name='Aggressive Banana Alter',
			signal=Harvest_Bananas_Sanction_Bananas()
		)
	)

	aggressive_apple_alter = Altar(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Altar_Properties(
			name='Aggressive Apple Alter',
			signal=Harvest_Apples_Sanction_Apples()
		)
	)

	peaceful_alter = Random_Alter(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Random_Altar_Properties(
			name='Peaceful Alter',
			signals=[Harvest_Apples_Sanction_Nothing(), Harvest_Bananas_Sanction_Nothing()]
		)
	)
	
	# This alter always says one thing
	# reliable_alter = Altar(
	# 	state=Entity_State(
	# 		position=Single_Point_Position()
	# 	),
	# 	properties=Altar_Properties(
	# 		name='Chieftain',
	# 		signal=Harvest_Apples_Sanction_Bananas()
	# 	)
	# )

	# Randomizes the sanctioning
	reliable_alter = Random_Alter(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Random_Altar_Properties(
			name='Chieftain',
			signals=[Harvest_Apples_Sanction_Bananas(), Harvest_Bananas_Sanction_Apples()]
		)
	)

	print('reliable_alter.properties:', reliable_alter.properties)

	enviornment_description = """Game Title: Alter Harvest

Objective: To accumulate the highest number of points over several rounds by strategically harvesting and sanctioning different fruits.

Gameplay: Players can imagine themselves being farmers within neighbouring nations. The game is played as a series of rounds. In each round, all players independently select a fruit to harvest and a fruit to sanction. At the end of each round reward/points are given based on the interactions between the resources chosen to be harvested and sanctions. There also exist alters which give information to players. This game is played for 1000 rounds."""

	num_obidient_agents = 4
	obidient_agents = []
	for i in range(num_obidient_agents):
		obidient_agents.append(Obedient_Harvest_Agent(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Entity_Properties(
				name=f'obedient agent {i + 1}'
			),
			master=reliable_alter
		))
	
	num_random_agents = 1
	random_agents = []
	for i in range(num_random_agents):
		random_agents.append(Random_Harvest_Agent(
			state=Entity_State(
			position=Single_Point_Position()
			),
			properties=Entity_Properties(
				name=f'random agent {i + 1}'
			),
		))

	explicit_belief_agent = Explicit_Belief_Harvest_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='explicit belief agent'
		),
		model=Human(),
		#model=ChatGPT(model_name='gpt-3.5-turbo', system_prompt='You are a strategic game playing agent.', verbosity=2),
		env_description=enviornment_description
	)

	altars = [aggressive_apple_alter, aggressive_banana_alter, peaceful_alter, reliable_alter]
	agents = obidient_agents + random_agents + [explicit_belief_agent]
	entities = altars + agents

	env = Simple_Altar_Env(
		state=Environment_State(
			entities=entities
		),
		properties=Environment_Properties(
			description=enviornment_description
		),
		reward_func=alter_reward_func
		#reward_func=zero_reward_func
	)
	
	run_sim(env=env, step_count=3)
