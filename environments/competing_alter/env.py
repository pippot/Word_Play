from word_play.environment import Action_Selection, Observation
from word_play.presets.environment_presets import Simple_Reset_Environment
from word_play.presets.observation_presets import format_possible_actions
from word_play.presets.movement_system_presets import SINGLE_POINT_MOVEMENT_SYSTEM
from dataclasses import dataclass
from environments.competing_alter.entities import Altar


@dataclass(slots=True)
class Alter_Env_Observation(Observation):
	last_reward: float
	alter_signals: list[tuple[str, str]]
	other_agent_actions_last_step: list[Action_Selection]
	other_agent_names: list[str]

	def __str__(self) -> str:
		obs = 'Alter Signals:'
		for signal in self.alter_signals:
			obs += f'\n{signal[0]} says: "{signal[1]}"'
		
		obs += "\n\nPrevious Player Actions:"
		if self.other_agent_actions_last_step:
			for agent_name, action_selection in zip(self.other_agent_names, self.other_agent_actions_last_step):
				obs += f'\n{agent_name}: {action_selection}'
		else:
			obs += "\nNo Previous Actions"

		obs += f'\n\nMy Reward: {self.last_reward}'
		obs += f'\n\n{format_possible_actions(self.possible_actions)}'
		return obs


@dataclass(slots=True)
class No_Alter_Observation(Observation):
	last_reward: float
	other_agent_actions_last_step: list[Action_Selection]
	other_agent_names: list[str]

	def __str__(self) -> str:
		obs = "Previous Player Actions:"
		if self.other_agent_actions_last_step:
			for agent_name, action_selection in zip(self.other_agent_names, self.other_agent_actions_last_step):
				obs += f'\n{agent_name}: {action_selection}'
		else:
			obs += "\nNo Previous Actions"

		obs += f'\n\nPrevious Round Reward: {self.last_reward}'
		obs += f'\n\n{format_possible_actions(self.possible_actions)}'
		return obs


class Simple_Altar_Env(Simple_Reset_Environment):

	movement_system = SINGLE_POINT_MOVEMENT_SYSTEM

	def _reset(self, seed=None) -> None:
		super()._reset(seed=seed)
		self.last_step_actions = []
		self.alter_signals: list[(str, str)] = []	# List of tuples (alter_name, signal)
		# Have alters output signals before the first step
		for alter in [entity for entity in self.state.entities if isinstance(entity, Altar)]:
			alter.step(env=self)

	def observe(self, agent_id: int) -> Alter_Env_Observation:
		return Alter_Env_Observation(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards[agent_id],
					alter_signals=self.alter_signals,
					other_agent_actions_last_step=[elm for idx, elm in enumerate(self.last_step_actions) if idx != agent_id],
					other_agent_names=[agent.properties.name for idx, agent in enumerate(self.agents) if idx != agent_id])

	def environment_start_of_step(self, action_selections: list[Action_Selection]):
		self.alter_signals = []	# Reset alter signals
		self.last_step_actions = action_selections
	
	def environment_end_of_step(self, action_selections: list[Action_Selection]):
		pass


class No_Alter_Obs_Env(Simple_Altar_Env):

	def observe(self, agent_id: int) -> No_Alter_Observation:
		return No_Alter_Observation(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards[agent_id],
					other_agent_actions_last_step=[elm for idx, elm in enumerate(self.last_step_actions) if idx != agent_id],
					other_agent_names=[agent.properties.name for idx, agent in enumerate(self.agents) if idx != agent_id])