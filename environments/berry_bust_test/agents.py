from environments.berry_bust_test.actions import PickBerry, Unlock
from word_play.presets.action_presets import Do_Nothing
from word_play.presets.agent_presets import Random_Action_Agent


#BERRY_PICKING_AGENT_EXPOSED_ACTIONS = (PickBerry(), Unlock())
BERRY_PICKING_AGENT_EXPOSED_ACTIONS = ()
BERRY_PICKING_AGENT_ACTIONS_ON_SELF = (Do_Nothing(),)

# class Berry_Picking_Agent(Entity):
# 	exposed_actions = (PickBerry(), Unlock())
# 	actions_on_self = (Do_Nothing())

class Random_Berry_Agent(Random_Action_Agent):
	exposed_actions = BERRY_PICKING_AGENT_EXPOSED_ACTIONS
	actions_on_self = BERRY_PICKING_AGENT_ACTIONS_ON_SELF