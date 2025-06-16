from word_play.environment import Action_Selection, Observation
from word_play.presets.observation_presets import format_possible_actions, format_discussion_phase
from dataclasses import dataclass
from environments.altar.common_entities import Altar_Signal


@dataclass(slots=True)
class Unreliable_Altar_Observation(Observation):
	last_reward: float
	# NOTE: we can filter out message from non-nearby agents when initializing the observation
	discussion_messages: list[list[tuple[int, str]]]
	discussion_phase_turn_count: int
	all_agent_names: list[str]
	all_altar_names: list[str]
	observing_agent_id: int
	agent_actions_last_turn: list[Action_Selection] | None
	altar_signals: list[Altar_Signal]
	cur_step: int
	all_potential_actions: list[Action_Selection]
	
	def __str__(self) -> str:
		raise NotImplementedError('Unreliable_Altar_Observation.__str__ not implemented yet')
		return 'STR METHOD NOT IMPLEMENTED YET'
		# TODO: this doesn't include the altar signal or agent actions
		# return f'Discussion Phase:\n{format_discussion_phase(self.discussion_messages, self.discussion_phase_turn_count, self.all_agent_names, self.observing_agent_id)}\n\nPrevious Round Reward: {self.last_reward}\n\nPossible Actions:\n{format_possible_actions(self.possible_actions)}'