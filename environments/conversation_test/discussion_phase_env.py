from word_play.environment import Action_Selection
from word_play.presets.environment_presets import Discussion_Phase_With_Reset_Environment
from word_play.presets.observation_presets import Discussion_Phase_With_Full_Info


class Conversation_Test_Env(Discussion_Phase_With_Reset_Environment):

	def observe(self, agent_id: int) -> Discussion_Phase_With_Full_Info:
		return Discussion_Phase_With_Full_Info(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards,
					# Note that get_entities_near_position() includes the agent itself
					discussion_messages=[
						[message for message in turn
	   						if self.movement_system.positions_are_close(
								self.agents[agent_id].state.position,
								self.agents[message.sender_id].state.position
							)
						] for turn in self.cur_discussion_messages
					],
					discussion_phase_turn_count=self.discussion_phase_turn_count,
					all_agent_names=[agent.properties.name for agent in self.agents],
					observing_agent_id=agent_id)
	
	def environment_start_of_step(self, action_selections: list[Action_Selection]):
		pass
	
	def environment_end_of_step(self, action_selections: list[Action_Selection]):
		pass