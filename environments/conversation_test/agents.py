from environments.conversation_test.actions import (
	Cook_A_Pie,
	Play_Poker,
	Exercise
)
from word_play.presets.action_presets import Do_Nothing
from word_play.presets.agent_presets import (
	Explicit_Belielf_Agent_With_Simple_Conversation,
	Explicit_Belief_Agent_With_Discussion_Phase
)


AGENT_EXPOSED_ACTIONS = ()
AGENT_ACTIONS_ON_SELF = (
	Cook_A_Pie(),
	Play_Poker(),
	Exercise(),
	Do_Nothing()
)


class Explicit_Belief_Conversation_Agent(Explicit_Belielf_Agent_With_Simple_Conversation):
	exposed_actions = AGENT_EXPOSED_ACTIONS
	actions_on_self = AGENT_ACTIONS_ON_SELF

class Explicit_Belief_Discussion_Phase_Agent(Explicit_Belief_Agent_With_Discussion_Phase):
	exposed_actions = AGENT_EXPOSED_ACTIONS
	actions_on_self = AGENT_ACTIONS_ON_SELF
