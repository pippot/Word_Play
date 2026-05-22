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
    "Trading_Policy",
    "sim_trade_negotiation",
    "Trade_Offer",
    "Trade_Session",
    "In_Active_Trade",
    "Can_Start_Trade",
    "Start_Trade",
    "Simple_Trading_Policy",
]
