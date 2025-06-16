from word_play.environment import Environment_State, Environment_Properties, Action_Selection, Environment, Observation
from word_play.presets.environment_presets import Simple_Reset_Environment
from word_play.presets.observation_presets import Possible_Actions_And_Last_Reward


class Berry_Patch_Env(Simple_Reset_Environment):

	def observe(self, agent_id: int) -> Possible_Actions_And_Last_Reward:
		return Possible_Actions_And_Last_Reward(
					possible_actions=self.get_possible_actions(agent_id),
					last_reward=self.last_rewards)
	
	def environment_start_of_step(self, action_selections: list[Action_Selection]):
		pass
	
	def environment_end_of_step(self, action_selections: list[Action_Selection]):
		pass