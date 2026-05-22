from word_play.presets.systems.communication.core import Communication_Policy
from word_play.presets.systems.communication.chat_room_action_communication.core import (
    Start_Private_Conversation,
    Start_Public_Conversation,
)
from word_play.presets.systems.communication.chat_room_action_communication.presets.policies import (
    Human_Communication_Policy,
    TalkingCow,
)
from word_play.presets.systems.communication.trade_communication.core import (
    Trading_Policy,
    sim_trade_negotiation,
)
from word_play.presets.systems.communication.trade_communication.trade_actions import (
    Trade_Offer,
    Trade_Session,
    In_Active_Trade,
    Can_Start_Trade,
    Start_Trade,
)


def __getattr__(name: str):
    if name == "Simple_Trading_Policy":
        from word_play.presets.systems.communication.trade_communication.presets.policies import (
            Simple_Trading_Policy,
        )

        return Simple_Trading_Policy
    raise AttributeError(name)

__all__ = [
    "Communication_Policy",
    "Start_Private_Conversation",
    "Start_Public_Conversation",
    "Human_Communication_Policy",
    "TalkingCow",
    "Trading_Policy",
    "sim_trade_negotiation",
    "Trade_Offer",
    "Trade_Session",
    "In_Active_Trade",
    "Can_Start_Trade",
    "Start_Trade",
    "Simple_Trading_Policy",
]
