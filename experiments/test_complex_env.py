from word_play.presets.movement_system_presets import Graph_Movement_System, positions_are_close_if_equal
from word_play.environment import Entity_State, Entity_Properties, Environment_State, Environment_Properties, Step_Execution_Order
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.model_presets import Human, ChatGPT, Llama3_Chat
from environments.altar.unreliable_altar.env import Unreliable_Altar_Env
from environments.altar.unreliable_altar.agents import (Simple_Agent, Normative_Agent, get_background_agent_system_prompt, \
	get_foreground_agent_system_prompt, anti_altar_select_action, altar_loving_select_action, no_belief_memory_select_action, \
	normative_select_action, Harvest_Agent_Properties, Harvest_Agent_State)
from environments.altar.common_entities import Altar, Altar_Properties, Fruit_Tree, Fruit_Tree_Properties, Allelopathic_Orchard, \
	Allelopathic_Orchard_Properties, Allelopathic_Orchard_State
from environments.altar.common_consts import Fruit_Type
from environments.altar.common_actions import Sanction_All_Nearby_Agents
from environments.altar.reward_funcs import allelopathic_reward
# TODO: dont import everything
from experiments.configs import *
from experiments.exp_setup import get_model, create_agent
from experiments.exp_exec import run_sim
import json


# TODO: move this to env creation file
def get_one_preferred_fruit_rewards(preferred_fruit: Fruit_Type, all_fruits: list[Fruit_Type]) -> dict[Fruit_Type, float]:
	return {fruit: 5 if fruit == preferred_fruit else 1 for fruit in all_fruits}


def get_fruits_types(num_fruits: int) -> list[Fruit_Type]:
	return list(Fruit_Type)[:num_fruits]


def main():
	node_names = ['the communal orchard', 'the residential area', 'the community centre']
	adj_matrix = [
		[0, 1, 1],
		[1, 0, 1],
		[1, 1, 0],
	]

	step_count = 4
	discussion_phase_turn_count = 3
	
	# graph_movement_sys, all_nodes = build_graph_movement_sys_from_adj_matrix(node_names, adj_matrix)
	graph_movement_sys = Graph_Movement_System(
		node_names=node_names,
		adjacency_matrix=adj_matrix,
		positions_are_close=positions_are_close_if_equal,
	)
	all_nodes = graph_movement_sys.all_nodes


	unique_fruit_count = 3
	fruits = get_fruits_types(unique_fruit_count)

	sanction_actions = [Sanction_All_Nearby_Agents(fruit) for fruit in fruits]
	Simple_Agent.actions_on_self += tuple(sanction_actions)

	# common_model_config = MED_TOP_P_LLAMA3_CHAT_8B_CONFIG
	# common_model_config = MED_TOP_P_LLAMA3_CHAT_70B_CONFIG
	common_model_config = HUMAN_MODEL_CONFIG

	alice = Simple_Agent(
		state=Harvest_Agent_State(
			position=all_nodes[0],
			most_recent_harvested_fruit=None
		),
		properties=Harvest_Agent_Properties(
			name='Alice',
			harvest_reward_per_fruit=get_one_preferred_fruit_rewards(Fruit_Type.APPLE, fruits)
		),
		model=get_model(
			common_model_config,
			'Alice',
			'background',
			2
		),
		discussion_prompt=ANTI_ALTAR_DISCUSSION_PROMPT_1_0,
		select_action_func=anti_altar_select_action,
	)

	bob = Simple_Agent(
		state=Harvest_Agent_State(
			position=all_nodes[0],
			most_recent_harvested_fruit=None
		),
		properties=Harvest_Agent_Properties(
			name='Bob',
			harvest_reward_per_fruit=get_one_preferred_fruit_rewards(Fruit_Type.BANANA, fruits)
		),
		model=get_model(
			common_model_config,
			'Bob',
			'background',
			2
		),
		discussion_prompt=ANTI_ALTAR_DISCUSSION_PROMPT_1_0,
		select_action_func=anti_altar_select_action,
	)

	andrei = Simple_Agent(
			state=Harvest_Agent_State(
				position=all_nodes[0],
				most_recent_harvested_fruit=None
			),
			properties=Harvest_Agent_Properties(
				name='Andrei',
				harvest_reward_per_fruit=get_one_preferred_fruit_rewards(Fruit_Type.BANANA, fruits)
			),
			model=get_model(
				common_model_config,
				'Andrei',
				'foreground',
				2
			),
			# discussion_prompt=NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0,
			discussion_prompt=ATRISHA_DISCUSSION_PROMPT_1_1,
			select_action_func=no_belief_memory_select_action,
		)


	altar = Altar(
		state=Entity_State(
			position=all_nodes[2]
		),
		properties=Altar_Properties(
			name='Chieftain Ophilia',
			signal=Fruit_Type.APPLE
		)
	)

	orchard = Allelopathic_Orchard(
		state=Allelopathic_Orchard_State(
			position=all_nodes[0],
			tree_counts=[0 for _ in range(len(fruits))],
			fruit_counts=[0 for _ in range(len(fruits))]
		),
		properties=Allelopathic_Orchard_Properties(
			name='the orchard',
			fruit_types=fruits,
			ripening_prob=0.9
		)
	)

	# fruit_trees = []
	# for fruit in fruits:
	# 	fruit_trees.append(Fruit_Tree(
	# 		state=Entity_State(
	# 			position=all_nodes[0]
	# 		),
	# 		properties=Fruit_Tree_Properties(
	# 			name=f'{fruit} tree',
	# 			fruit=str(fruit)
	# 		)
	# 	))
	

	env = Unreliable_Altar_Env(
		state=Environment_State(
			# entities=[alice, bob, andrei, altar] + fruit_trees,
			entities=[alice, bob, andrei, altar, orchard],
		),
		properties=Environment_Properties(
			description='In this environment all community members are part of a clan.'
		),
		movement_system=graph_movement_sys,
		reward_func=allelopathic_reward,
		step_execution_order=Step_Execution_Order.Entity_Definition_Order,
		discussion_phase_turn_count=discussion_phase_turn_count,
	)

	results = run_sim(env=env, step_count=step_count)

	complete_exp_info = {
		'exp_config': {
			'discussion_phase_turn_count': discussion_phase_turn_count,
			'step_count': step_count,
		},
		# 'completion_time': completion_time,
		# 'exp_duration_in_secs': exp_duration_in_secs,
		'agent_names': ['Alice', 'Bob', 'Andrei'],
		'results': results
	}
	
	with open('/h/andrei/normative_agents/experiments/test_complex_env_res_A.json', 'w', encoding='utf-8') as f:
		json.dump(complete_exp_info, f, indent=4, default=str)


if __name__	 == '__main__':
	main()