from word_play.environment import Entity_State, Entity_Properties, Environment_State, Environment_Properties, Step_Execution_Order
from word_play.presets.movement_system_presets import Single_Point_Position, SINGLE_POINT_MOVEMENT_SYSTEM
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.model_presets import Human, ChatGPT, Llama3_Chat
from environments.altar.unreliable_altar.env import Unreliable_Altar_Env
from environments.altar.unreliable_altar.agents import (Simple_Agent, Normative_Agent, get_background_agent_system_prompt, \
	get_foreground_agent_system_prompt, get_normatively_prompted_agent_system_prompt, anti_altar_select_action, altar_loving_select_action, no_belief_memory_select_action, normative_select_action)
from environments.altar.common_entities import Altar, Altar_Properties, Altar_Signal_Type, Fruit_Tree, Fruit_Tree_Properties
# TODO: dont import everything
from experiments.configs import *
import copy
import random


# TODO: this enum stuff is kinda annoying
FRUIT_STR_TO_ENUM = {
	'apple': Altar_Signal_Type.APPLE,
	'banana': Altar_Signal_Type.BANANA,
	'peach': Altar_Signal_Type.PEACH,
	'orange': Altar_Signal_Type.ORANGE,
	'plum': Altar_Signal_Type.PLUM,
}


def get_model(model_config: dict, agent_name: str, agent_type: str, verbosity: int):
	if agent_type == 'background':
		system_prompt = get_background_agent_system_prompt(agent_name)
	elif agent_type == 'foreground':
		system_prompt = get_foreground_agent_system_prompt(agent_name)
		# system_prompt = get_normatively_prompted_agent_system_prompt(agent_name)
	else:
		raise ValueError(f"Agent type {agent_type} is not supported.")
	
	if model_config['model_type'] == 'Human':
		return Human()
	elif model_config['model_type'] == 'ChatGPT':
		return ChatGPT(
			model_name=model_config['model_name'],
			system_prompt=system_prompt,
			model_params=model_config['model_params'],
			verbosity=verbosity)
	elif model_config['model_type'] == 'Llama3_Chat':
		return Llama3_Chat(
			model_name=model_config['model_name'],
			system_prompt=system_prompt,
			model_params=model_config['model_params'],
			verbosity=verbosity)
	else:
		raise ValueError(f"Model name {model_config['model_type']} is not supported.")


def create_agent(agent_config: dict, agent_name: str, agent_type: str, verbosity: int):
	if agent_config['agent_type'] == 'no_belief_memory':
		return Simple_Agent(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Entity_Properties(
				name=agent_name,
			),
			model=get_model(
				agent_config['model_config'],
				agent_name,
				agent_type,
				verbosity
			),
			discussion_prompt=agent_config['discussion_prompt'],
			select_action_func=no_belief_memory_select_action
		)
	elif agent_config['agent_type'] == 'normative':
		if 'kwargs' in agent_config:
			kwargs = agent_config['kwargs']
		else:
			kwargs = {}
		return Normative_Agent(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Entity_Properties(
				name=agent_name,
			),
			model=get_model(
				agent_config['model_config'],
				agent_name,
				agent_type,
				verbosity
			),
			discussion_prompt=agent_config['discussion_prompt'],
			select_action_func=normative_select_action,
			**kwargs
		)
	else:
		raise ValueError(f"Agent name {agent_config['model_type']} is not supported.")


def create_simple_env(
			# TODO: should likely properly define a config for foreground agents
			foreground_agent_configs: list[dict],

			num_anti_altar_agents: int,
			anti_altar_agent_prompt: str,
			anti_altar_agent_model_config: dict,
			
			num_altar_loving_agents: int,
			altar_loving_agent_prompt: str,
			altar_loving_agent_model_config: dict,
			
			fruit_tree_types: list[str],
			altar_fruit_types: list[str],

			discussion_phase_turn_count: int,
			# game_description: str,	# I don't have a nice use for this yet
			# TODO: should make some args for the altar
			
			verbosity: int = 0,
			# NOTE: we can add more parameters here as needed
		) -> Unreliable_Altar_Env:

	assert num_anti_altar_agents + num_altar_loving_agents + len(altar_fruit_types) < len(LIST_OF_NAMES), "Not enough names"
	# apples and banana are currently hardcoded
	assert len(fruit_tree_types) >= 2 and 'apple' in fruit_tree_types and 'banana' in fruit_tree_types, "apple and banana are hardcoded"

	# Creating the agents
	agents = []
	all_names = copy.deepcopy(LIST_OF_NAMES)

	# Foreground agents
	for agent_config in foreground_agent_configs:
		# NOTE: we can shuffle the name list if we want to randomize the agent names
		agent_name = all_names.pop(0)
		agents.append(create_agent(agent_config, agent_name, 'foreground', verbosity))

	
	# Anti Altar Agents
	for _ in range(num_anti_altar_agents):
		# NOTE: we can shuffle the name list if we want to randomize the agent names
		agent_name = all_names.pop(0)
		agents.append(Simple_Agent(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Entity_Properties(
				name=agent_name,
			),
			model=get_model(
				anti_altar_agent_model_config,
				agent_name,
				'background',
				verbosity
			),
			discussion_prompt=anti_altar_agent_prompt,
			select_action_func=anti_altar_select_action
		))

	# Altar Loving Agents
	for _ in range(num_altar_loving_agents):
		# NOTE: we can shuffle the name list if we want to randomize the agent names
		agent_name = all_names.pop(0)
		agents.append(Simple_Agent(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Entity_Properties(
				name=agent_name,
			),
			model=get_model(
				altar_loving_agent_model_config,
				agent_name,
				'background',
				verbosity
			),
			discussion_prompt=altar_loving_agent_prompt,
			select_action_func=altar_loving_select_action
		))

	# Randomize agent action order (ex., we don't anti alter agents always speaking first)	
	random.shuffle(agents)
	agent_names = [agent.properties.name for agent in agents]
	

	# Creating non-agent entities
	non_agent_entities = []

	# TODO: would be nice to merge the name lists. We did this because we hard coded an altar name in some of the prompts
	all_possible_altar_names = LIST_OF_ALTAR_NAMES

	# Altar
	for fruit_name in altar_fruit_types:
		altar_name = all_possible_altar_names.pop(0)
		non_agent_entities.append(Altar(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Altar_Properties(
				name=f'Chieftain {altar_name}',
				signal=FRUIT_STR_TO_ENUM[fruit_name]
			)
		))

	# Fruit Trees
	for fruit_name in fruit_tree_types:
		non_agent_entities.append(Fruit_Tree(
			state=Entity_State(
				position=Single_Point_Position()
			),
			properties=Fruit_Tree_Properties(
				name=f'{fruit_name} tree',
				fruit=fruit_name
			)
		))

	
	# Creating the environment
	return Unreliable_Altar_Env(
		state=Environment_State(
			entities=agents + non_agent_entities,
		),
		properties=Environment_Properties(
			description='In this environment all community members are part of a clan lead by chief.'
		),
		movement_system=SINGLE_POINT_MOVEMENT_SYSTEM,
		reward_func=zero_reward_func,
		step_execution_order=Step_Execution_Order.Entity_Definition_Order,
		discussion_phase_turn_count=discussion_phase_turn_count,
	), agent_names

