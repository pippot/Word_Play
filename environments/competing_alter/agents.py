from environments.alter_common.actions import (
	Harvest_Apples_Sanction_Nothing,
	Harvest_Apples_Sanction_Apples,
	Harvest_Apples_Sanction_Bananas,
	Harvest_Bananas_Sanction_Nothing,
	Harvest_Bananas_Sanction_Apples,
	Harvest_Bananas_Sanction_Bananas,
)

from word_play.presets.agent_presets import (
	Explicit_Belief_Agent,
	extract_number_surrounded_by_quotes
)
from word_play.presets.observation_presets import format_possible_actions
from word_play.presets.agent_presets import Random_Action_Agent
from word_play.environment import Observation, Agent, Action_Selection, Action, Entity, Entity_State, Entity_Properties

AGENT_EXPOSED_ACTIONS = ()
AGENT_ACTIONS_ON_SELF = (
	Harvest_Apples_Sanction_Bananas(),
	Harvest_Apples_Sanction_Apples(),
	Harvest_Apples_Sanction_Nothing(),
	Harvest_Bananas_Sanction_Bananas(),
	Harvest_Bananas_Sanction_Apples(),
	Harvest_Bananas_Sanction_Nothing()
)

class Random_Harvest_Agent(Random_Action_Agent):
	exposed_actions = AGENT_EXPOSED_ACTIONS
	actions_on_self = AGENT_ACTIONS_ON_SELF

class Explicit_Belief_Harvest_Agent(Explicit_Belief_Agent):
	exposed_actions = AGENT_EXPOSED_ACTIONS
	actions_on_self = AGENT_ACTIONS_ON_SELF

	# Overwriting the history_to_str method
	def history_to_str(self) -> str:
		history_str = f''
		# iterate over the last history_length rounds
		for round, (obs, my_action) in self.get_history_enumerator():
			if not obs.other_agent_actions_last_step:
				all_agent_actions = ['None' for agent in obs.other_agent_names] + [my_action]
			else:
				all_agent_actions = obs.other_agent_actions_last_step + [my_action]

			all_agent_names = obs.other_agent_names + ['Me']
			all_actions_str = ''
			for agent_name, action_selection in zip(all_agent_names, all_agent_actions):
				all_actions_str += f'\n{agent_name}: {action_selection}'
			
			history_str += f'\n\n## Round {round}:\n### Actions:{all_actions_str}\n### My Reward: {obs.last_reward}'

		if not history_str:
			history_str = 'No prior history.'
		
		return history_str.strip()
	
	# Overwriting ask_model_to_select_action method
	def ask_model_to_select_action(self) -> tuple[int, str]:
		prompt = f"""# Game Description:
{self.env_description}

# History:
{self.history_to_str()}

# Expectations:
{self.expectations}

# {format_possible_actions(self.obs_history[-1].possible_actions)}

# Current Round:
What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, output the index of your final action selection surrounded by quotation marks (e.g. Final Answer: "<action_selection>"):"""
		
		selection_CoT = self.model.generate_text(prompt).strip()
		action_idx = extract_number_surrounded_by_quotes(selection_CoT)
		return action_idx, selection_CoT, prompt

class Obedient_Harvest_Agent(Agent):
	exposed_actions = AGENT_EXPOSED_ACTIONS
	actions_on_self = AGENT_ACTIONS_ON_SELF
	
	def __init__(self, state: Entity_State, properties: Entity_Properties, master: Entity) -> None:
		super().__init__(state=state, properties=properties)
		self.master = master
	
	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		action_selection = Action_Selection(
			action=self.master.get_signal(),
			target_entity=self)
		return action_selection, {}
	
	def step(self):
		pass
		



