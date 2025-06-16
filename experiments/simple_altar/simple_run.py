from word_play.environment import Environment_State, Environment_Properties, Entity_State, Entity_Properties, Environment
from environments.altar.simple_altar.env import Simple_Altar_Env
from environments.altar.simple_altar.entities import Altar, Altar_Properties
from environments.altar.simple_altar.agents import Random_Harvest_Agent, Explicit_Belief_Harvest_Agent
from environments.altar.simple_altar.reward_func import altar_reward_func
from word_play.presets.model_presets import Human, ChatGPT
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.movement_system_presets import Single_Point_Position, SINGLE_POINT_MOVEMENT_SYSTEM


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
			# print('observation.altar_signals:', observation.altar_signals)
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
	banana_altar = Altar(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Altar_Properties(
			name='Banana Altar',
			signal='Eat bananas!'
		)
	)
	apple_altar = Altar(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Altar_Properties(
			name='Apple Altar',
			signal='Eat apples!'
		)
	)
	print('banana_altar.properties:', banana_altar.properties)

	enviornment_description = """Game Title: Altar Harvest

Objective: To accumulate the highest number of points over several rounds by strategically harvesting and sanctioning different fruits.

Gameplay: Players can imagine themselves being farmers within neighbouring nations. The game is played as a series of rounds. In each round, all players independently select a fruit to harvest and a fruit to sanction. At the end of each round reward/points are given based on the interactions between the resources chosen to be harvested and sanctions. There also exist altars which give information to players. This game is played for 1000 rounds."""

	random_agent_1 = Random_Harvest_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='random agent 1'
		)
	)
	random_agent_2 = Random_Harvest_Agent(
		state=Entity_State(
			position=Single_Point_Position()
		),
		properties=Entity_Properties(
			name='random agent 2'
		)
	)
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

	env = Simple_Altar_Env(
		state=Environment_State(
			entities=[banana_altar, apple_altar, random_agent_1, random_agent_2, explicit_belief_agent]
		),
		properties=Environment_Properties(
			description=enviornment_description
		),
		movement_system=SINGLE_POINT_MOVEMENT_SYSTEM,
		reward_func=altar_reward_func
		#reward_func=zero_reward_func
	)
	
	run_sim(env=env, step_count=3)
