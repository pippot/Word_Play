from word_play.environment import Action_Selection, Observation
from word_play.presets.environment_presets import Simple_Reset_Environment
from word_play.presets.observation_presets import format_possible_actions
from dataclasses import dataclass
from environments.altar.simple_altar.entities import Altar


@dataclass(slots=True)
class Altar_Env_Observation(Observation):
	last_reward: float
	altar_signals: list[tuple[str, str]]
	other_agent_actions_last_step: list[Action_Selection]
	other_all_agent_names: list[str]

	def __str__(self) -> str:
		obs = 'Altar Signals:'
		for signal in self.altar_signals:
			obs += f'\n{signal[0]} says: "{signal[1]}"'
		
		obs += "\n\nPrevious Player Actions:"
		if self.other_agent_actions_last_step:
			for agent_name, action_selection in zip(self.other_all_agent_names, self.other_agent_actions_last_step):
				obs += f'\n{agent_name}: {action_selection}'
		else:
			obs += "\nNo Previous Actions"

		obs += f'\n\nMy Reward: {self.last_reward}'
		obs += f'\n\nPossible Actions:\n{format_possible_actions(self.possible_actions)}'
		return obs


@dataclass(slots=True)
class No_Altar_Observation(Observation):
	last_reward: float
	other_agent_actions_last_step: list[Action_Selection]
	other_all_agent_names: list[str]

	def __str__(self) -> str:
		obs = "Previous Player Actions:"
		if self.other_agent_actions_last_step:
			for agent_name, action_selection in zip(self.other_all_agent_names, self.other_agent_actions_last_step):
				obs += f'\n{agent_name}: {action_selection}'
		else:
			obs += "\nNo Previous Actions"

		obs += f'\n\nPrevious Round Reward: {self.last_reward}'
		obs += f'\n\nPossible Actions:\n{format_possible_actions(self.possible_actions)}'
		return obs


class Simple_Altar_Env(Simple_Reset_Environment):

	def _reset(self, seed=None) -> None:
		super()._reset(seed=seed)
		self.last_step_actions = []
		self.altar_signals: list[(str, str)] = []	# List of tuples (altar_name, signal)
		# Have altars output signals before the first step
		for altar in [entity for entity in self.state.entities if isinstance(entity, Altar)]:
			altar.step(env=self)

	def observe(self, agent_id: int) -> Altar_Env_Observation:
		return Altar_Env_Observation(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards[agent_id],
					altar_signals=self.altar_signals,
					other_agent_actions_last_step=[elm for idx, elm in enumerate(self.last_step_actions) if idx != agent_id],
					other_all_agent_names=[agent.properties.name for idx, agent in enumerate(self.agents) if idx != agent_id])

	def environment_start_of_step(self, action_selections: list[Action_Selection]):
		self.altar_signals = []	# Reset altar signals
		self.last_step_actions = action_selections
	
	def environment_end_of_step(self, action_selections: list[Action_Selection]):
		pass


class No_Altar_Obs_Env(Simple_Altar_Env):

	def observe(self, agent_id: int) -> No_Altar_Observation:
		return No_Altar_Observation(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards[agent_id],
					other_agent_actions_last_step=[elm for idx, elm in enumerate(self.last_step_actions) if idx != agent_id],
					other_all_agent_names=[agent.properties.name for idx, agent in enumerate(self.agents) if idx != agent_id])