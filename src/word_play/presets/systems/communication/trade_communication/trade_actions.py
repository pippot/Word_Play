"""Trade actions — Trade_Session state, Start_Trade action, and validations.

Mirrors chat_room_action_communication/core.py which has Start_Public_Conversation
and sim_simple_conversation. Here we have Start_Trade and sim_trade_negotiation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from word_play.core import Component, Entity, Environment, Action, Action_Validation
from word_play.core.actions import Target_Is_Self, Target_Is_Nearby, Target_Not_Self
from word_play.presets.systems.inventory import Inventory
from word_play.presets.systems.currency import Money
from word_play.presets.renderers.renderer import Renderable


@dataclass
class Trade_Offer:
    items: list = field(default_factory=list)
    currency: int = 0
    accepted: bool = False


class Trade_Session(Component):
    """Trade session attached to both participants during trade.

    Agents call methods directly on this component:
    - set_offer(entity, items, currency) -> dict
    - accept(entity) -> dict
    - decline(entity) -> dict
    - view(viewer) -> dict
    """

    def __init__(self, a: Entity, b: Entity):
        super().__init__()
        self.partners = (a, b)
        self.offers = {a: Trade_Offer(), b: Trade_Offer()}
        self._completed = False

    def set_offer(self, entity: Entity, items: list = None, currency: int = 0) -> dict:
        """Set entity's offer. Returns previous offer to inventory."""
        self._refund(entity, self.offers[entity])
        for offer in self.offers.values():
            offer.accepted = False
        self.offers[entity] = Trade_Offer(items or [], currency)
        self._reserve(entity, self.offers[entity])
        self._update_renderables()
        return {"offer_set": True}

    def _reserve(self, entity: Entity, offer: Trade_Offer) -> None:
        inv = entity.get_component(Inventory)
        money = entity.get_component(Money)
        if inv:
            for i in offer.items:
                if i in inv.contents:
                    inv.contents.remove(i)
        if money and offer.currency:
            money.subtract(offer.currency)

    def _refund(self, entity: Entity, offer: Trade_Offer) -> None:
        inv = entity.get_component(Inventory)
        money = entity.get_component(Money)
        if inv:
            for i in offer.items:
                i.position = entity.position
                inv.contents.append(i)
        if money and offer.currency:
            money.add(offer.currency)

    def accept(self, entity: Entity) -> dict:
        """Entity accepts trade. Executes if both accept."""
        if self._completed:
            return {"error": "trade already completed"}
        self.offers[entity].accepted = True
        self._update_renderables()
        if all(o.accepted for o in self.offers.values()):
            self._completed = True
            self._refund(self.partners[0], self.offers[self.partners[1]])
            self._refund(self.partners[1], self.offers[self.partners[0]])
            self.close()
            return {"executed": True}
        return {"waiting": True}

    def decline(self, entity: Entity) -> dict:
        """Cancel trade - refund all offers."""
        if self._completed:
            return {"error": "already completed"}
        for p, offer in self.offers.items():
            self._refund(p, offer)
        self.close()
        return {"declined": True}

    def close(self) -> None:
        for p in self.partners:
            p.components.pop(Trade_Session, None)

    def partner(self, entity: Entity) -> Entity:
        """Get the other participant."""
        a, b = self.partners
        return b if entity is a else a

    def _update_renderables(self) -> None:
        """Update last_message on Renderable for both participants."""
        for party in self.partners:
            renderable = party.get_component(Renderable)
            if renderable:
                other = self.partner(party)
                your_items = ",".join([i.name for i in self.offers[party].items]) if self.offers[party].items else "none"
                their_items = ",".join([i.name for i in self.offers[other].items]) if self.offers[other].items else "none"
                msg = f"TRADE:|you_offer:{your_items}|you_currency:{self.offers[party].currency}|they_offer:{their_items}|they_currency:{self.offers[other].currency}|you_accepted:{'yes' if self.offers[party].accepted else 'no'}|they_accepted:{'yes' if self.offers[other].accepted else 'no'}|partner:{other.name}"
                renderable.last_message = msg

    def view(self, viewer: Entity) -> dict:
        other = self.partner(viewer)
        return {
            "you": {"items": [i.name for i in self.offers[viewer].items], "currency": self.offers[viewer].currency, "accepted": self.offers[viewer].accepted},
            "them": {"items": [i.name for i in self.offers[other].items], "currency": self.offers[other].currency, "accepted": self.offers[other].accepted},
        }


class In_Active_Trade(Action_Validation):
    def is_valid(self, actor: Entity, target: Entity, env: Environment) -> bool:
        return Trade_Session in actor.components


class Can_Start_Trade(Action_Validation):
    def is_valid(self, actor: Entity, target: Entity, env: Environment) -> bool:
        return Trade_Session not in actor.components and Trade_Session not in target.components


class Start_Trade(Action):
    """Start trade with nearby entity.

    Creates a Trade_Session, then runs sim_trade_negotiation if both
    parties have a Trading_Policy. If not, just creates the session
    and updates renderables (for manual/LLM-driven negotiation).
    """

    def __init__(self):
        super().__init__(validation_rules=[Target_Not_Self(), Target_Is_Nearby(), Can_Start_Trade()])

    def exec_action(self, actor: Entity, target: Entity, env: Environment, kwargs: dict | None) -> dict:
        from word_play.presets.systems.communication.trade_communication.core import (
            Trading_Policy,
            sim_trade_negotiation,
        )

        session = Trade_Session(actor, target)
        actor.components[Trade_Session] = target.components[Trade_Session] = session
        session._update_renderables()

        if actor.get_component(Trading_Policy) and target.get_component(Trading_Policy):
            return sim_trade_negotiation(session, env)

        return {"trade_started": True, "with": target.name}

    def action_description_text(self, actor: Entity, target: Entity, env: Environment) -> str:
        return f"Start trade with {target.name}"


__all__ = [
    "Trade_Offer",
    "Trade_Session",
    "In_Active_Trade",
    "Can_Start_Trade",
    "Start_Trade",
]
