from word_play.environment import Action_Selection, Environment
from environments.alter_common.actions import *  # "import *" is not ideal...


REWARD_FOR_APPLE_HARVEST = 9
REWARD_FOR_BANANA_HARVEST = 10
REWARD_FOR_SANCTIONER = 5
REWARD_FOR_SANCTIONEE = -20


def alter_reward_func(
	agent_actions: list[Action_Selection], env: Environment
) -> list[float]:

	reward = [0] * len(env.agents)

	apple_harvest_count = sum(1 for action, _ in agent_actions if action.harvest == "apples")
	banana_harvest_count = sum(1 for action, _ in agent_actions if action.harvest == "bananas")
	apple_sanction_count = sum(1 for action, _ in agent_actions if action.sanction == "apples")
	banana_sanction_count = sum(1 for action, _ in agent_actions if action.sanction == "bananas")
   
	for agent_id, (action, target_entity) in enumerate(agent_actions):
		cur_agent_apple_harvest = 1 if action.harvest == 'apples' else 0
		cur_agent_banana_harvest = 1 if action.harvest == 'bananas' else 0
		cur_agent_apple_sanction = 1 if action.sanction == 'apples' else 0
		cur_agent_banana_sanction = 1 if action.sanction == 'bananas' else 0

		if action.harvest == "apples":
			# harvesting reward
			reward[agent_id] += REWARD_FOR_APPLE_HARVEST
			# sanctionee reward
			reward[agent_id] += REWARD_FOR_SANCTIONEE * (apple_sanction_count - cur_agent_apple_sanction)
		elif action.harvest == "bananas":
			# harvesting reward
			reward[agent_id] += REWARD_FOR_BANANA_HARVEST
			# sanctionee reward
			reward[agent_id] += REWARD_FOR_SANCTIONEE * (banana_sanction_count - cur_agent_banana_sanction)
		else:
			raise ValueError(f"Unknown harvest type: {action.harvest}")

		# sanctioning reward
		if action.sanction == "apples":
			reward[agent_id] += REWARD_FOR_SANCTIONER * max(0, apple_harvest_count - cur_agent_apple_harvest)
		elif action.sanction == "bananas":
			reward[agent_id] += REWARD_FOR_SANCTIONER * max(0, banana_harvest_count - cur_agent_banana_harvest)
		elif action.sanction == "nothing":
			pass
		else:
			raise ValueError(f"Unknown sanction type: {action.sanction}")

	return reward