"""Systems - reusable game systems for Word Play.

Each module provides a self-contained system:
- inventory: Item storage and movement
- containers: Storage containers and infinite sources
- crafter: Crafting recipes and stations
- currency: Wallet and currency transfer
- trading: Trade sessions and negotiation
- delivery: Order fulfillment
- combat: Attack
- health: HP tracking and damage/death
- ownership: Claim and give ownership
- action_compositions: Action chaining with per-step validation
- do_nothing: No-op action
- regrowable: Consumable entities that reappear on a timer
- freezable: Temporarily hold an entity in place
- destructible: Entities that transform when their HP hits zero
- team_marker: Team identification and ally/enemy validators
"""
from word_play.presets.systems.action_compositions import (
    Action_Comp,
    Step, Required, Optional,
)
from word_play.presets.systems.combat import Attack
from word_play.presets.systems.containers import (
    Container,
    Open_Container,
    Single_Item_Holder,
    Regrowable_Item_Source,
    Regen_Pool,
    Take_From_Infinite_Source,
    Take_From_Regen_Pool,
    Clean_Pool,
)
from word_play.presets.systems.crafter import Crafter, Crafter_Recipe, Load_Crafter, Load_First_Into_Crafter, Collect_From_Crafter, Zap_Change, Zap_Change_Inventory
from word_play.presets.systems.currency import (
    Money,
    Has_Money,
    Has_Currency,
    Currency_Amount_Arg,
)
from word_play.presets.systems.delivery import (
    Deliver_Item,
    Front_Order_Match,
    Named_Item_Reward,
    Record_Delivery,
)
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import (
    Inventory,
    Inventory_Item_Index_Arg,
    Target_Not_In_Inventory,
    Pick_Up_Item,
    Put_In_Container,
    Drop_Item,
)
from word_play.presets.systems.ownership import (
    Owner,
    Ownable,
    Claim,
    Give_Ownership,
    Target_Is_Unowned,
)
from word_play.presets.systems.communication.trade_communication.trade_actions import (
    Trade_Session,
    Trade_Offer,
    In_Active_Trade,
    Can_Start_Trade,
    Start_Trade,
)
from word_play.presets.systems.communication.trade_communication.core import (
    Trading_Policy,
    sim_trade_negotiation,
)
from word_play.presets.systems.communication.trade_communication.presets.policies import (
    Simple_Trading_Policy,
)
from word_play.presets.systems.regrowable import (
    Regrowable,
    Consume_Regrowable,
    Harvest_Regrowable,
)
from word_play.presets.systems.freezable import (
    Freezable,
    Freeze,
)
from word_play.presets.systems.destructible import Destructible
from word_play.presets.systems.team_marker import (
    Team,
    Target_Is_Enemy,
    Target_Is_Ally,
)
from word_play.presets.systems.preferences import Preference
from word_play.presets.systems.reward import Rewardable, award_reward
from word_play.presets.systems.cooldown import Cooldown, Action_On_Cooldown

__all__ = [
    # Action compositions
    "Action_Comp",
    "Step",
    "Required",
    "Optional",
    # Combat
    "Attack",
    # Health
    "Health",
    # Containers
    "Container",
    "Open_Container",
    "Single_Item_Holder",
    "Regrowable_Item_Source",
    "Regen_Pool",
    "Take_From_Infinite_Source",
    "Take_From_Regen_Pool",
    "Clean_Pool",
    # Crafter
    "Crafter",
    "Crafter_Recipe",
    "Load_Crafter",
    "Load_First_Into_Crafter",
    "Collect_From_Crafter",
    "Zap_Change",
    "Zap_Change_Inventory",
    # Currency
    "Money",
    "Has_Money",
    "Has_Currency",
    "Currency_Amount_Arg",
    # Delivery
    "Deliver_Item",
    "Front_Order_Match",
    "Named_Item_Reward",
    "Record_Delivery",
    # Do Nothing
    "Do_Nothing",
    # Inventory
    "Inventory",
    "Inventory_Item_Index_Arg",
    "Target_Not_In_Inventory",
    "Pick_Up_Item",
    "Put_In_Container",
    "Drop_Item",
    # Trading
    "Trade_Session",
    "Trade_Offer",
    "In_Active_Trade",
    "Can_Start_Trade",
    "Start_Trade",
    "Trading_Policy",
    "sim_trade_negotiation",
    "Simple_Trading_Policy",
    # Ownership
    "Owner",
    "Ownable",
    "Claim",
    "Give_Ownership",
    "Target_Is_Unowned",
    # Regrowable
    "Regrowable",
    "Consume_Regrowable",
    "Harvest_Regrowable",
    # Freezable
    "Freezable",
    "Freeze",
    # Destructible
    "Destructible",
    # Team marker
    "Team",
    "Target_Is_Enemy",
    "Target_Is_Ally",
    # Preferences
    "Preference",
    # Reward
    "Rewardable",
    "award_reward",
    # Cooldown
    "Cooldown",
    "Action_On_Cooldown",
]
