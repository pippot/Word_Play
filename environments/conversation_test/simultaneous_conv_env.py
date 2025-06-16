from word_play.environment import Action_Selection
from word_play.presets.environment_presets import Conversation_And_Reset_Environment
from word_play.presets.observation_presets import Conversation_Possible_Actions_And_Last_Reward


class Conversation_Test_Env(Conversation_And_Reset_Environment):

	def observe(self, agent_id: int) -> Conversation_Possible_Actions_And_Last_Reward:
		return Conversation_Possible_Actions_And_Last_Reward(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards,
					# Note that get_entities_near_position() includes the agent itself
					conversation=[self.conversation[self.agent_to_idx[agent]] for agent in self.get_entities_near_position(self.agents[agent_id].state.position)],
					all_agent_names=[agent.properties.name for agent in self.agents],
					observing_agent_id=agent_id)
	
	def environment_start_of_step(self, action_selections: list[Action_Selection]):
		pass
	
	def environment_end_of_step(self, action_selections: list[Action_Selection]):
		pass