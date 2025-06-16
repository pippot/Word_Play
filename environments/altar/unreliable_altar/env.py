from word_play.environment import Observation, Action_Selection, Agent
from word_play.presets.environment_presets import Discussion_Phase_With_Reset_Environment
from environments.altar.unreliable_altar.observations import Unreliable_Altar_Observation
from environments.altar.common_entities import Altar, Altar_Signal


class Unreliable_Altar_Env(Discussion_Phase_With_Reset_Environment):
	def _reset(self, seed=None) -> None:
		super()._reset(seed=seed)
		self.last_step_actions = []
		self.altar_signals: list[Altar_Signal] = []
		# Have altars output signals before the first step
		for altar in [entity for entity in self.state.entities if isinstance(entity, Altar)]:
			altar.step(env=self)
		self.cur_step = 0
	
	
	def observe(self, agent_id: int) -> Observation:
		# nearby_agent_ids includes the observing agent
		nearby_agent_ids = [self.agent_to_idx[entity] for entity in self.get_entities_near_position(self.agents[agent_id].state.position)
				   			if isinstance(entity, Agent)]
		
		agent = self.agents[agent_id]
		all_potential_actions = [Action_Selection(action=action, target_entity=agent) for action in agent.actions_on_self]
		for entity in self.state.entities:
			all_potential_actions += [Action_Selection(action=action, target_entity=entity) for action in entity.exposed_actions]

		return Unreliable_Altar_Observation(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards[agent_id],
					# filter out message from non-nearby agents
					discussion_messages=[[message for message in turn if message.sender_id in nearby_agent_ids] 
											for turn in self.cur_discussion_messages],
					discussion_phase_turn_count=self.discussion_phase_turn_count,
					all_agent_names=[agent.properties.name for agent in self.agents],
					all_altar_names=[altar.properties.name for altar in self.state.entities if isinstance(altar, Altar)],
					observing_agent_id=agent_id,
					agent_actions_last_turn=self.last_step_actions,
					altar_signals=self.altar_signals,
					cur_step=self.cur_step,
					all_potential_actions=all_potential_actions,
				)


	def environment_start_of_step(self, action_selections: list[Action_Selection]):
		self.altar_signals = []	# Reset altar signals
		self.last_step_actions = action_selections


	def environment_end_of_step(self, action_selections: list[Action_Selection]):
		self.cur_step += 1