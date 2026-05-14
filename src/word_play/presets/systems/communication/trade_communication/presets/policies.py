"""Concrete Trading_Policy implementations."""
from __future__ import annotations

from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.systems.communication.trade_communication.core import Trading_Policy
from word_play.presets.systems.inventory import Inventory
from word_play.presets.systems.currency import Money


class Simple_Trading_Policy(LLM_Action_And_Communication_Policy, Trading_Policy):
    """Combined action, communication, and trading policy backed by any Model.

    Use this when an agent needs to participate in trade sessions.
    LLM_Action_And_Communication_Policy alone does NOT grant trading ability.
    """

    def propose_trade(self, session, entity, env):
        inv = entity.get_component(Inventory) if entity.get_component(Inventory) else None
        money = entity.get_component(Money) if entity.get_component(Money) else None
        view = session.view(entity)
        partner = session.partner(entity)
        partner_name = partner.name
        partner_inv = partner.get_component(Inventory) if partner.get_component(Inventory) else None
        partner_money = partner.get_component(Money) if partner.get_component(Money) else None

        inv_items = []
        if inv:
            inv_items = [f"{i}: {item.name}" for i, item in enumerate(inv.contents)]
        inv_str = ", ".join(inv_items) if inv_items else "empty"
        money_str = f"{money.amount} {money.currency_name}" if money else "none"

        partner_inv_items = []
        if partner_inv:
            partner_inv_items = [item.name for item in partner_inv.contents]
        partner_inv_str = ", ".join(partner_inv_items) if partner_inv_items else "empty"
        partner_money_str = f"{partner_money.amount} {partner_money.currency_name}" if partner_money else "none"

        round_context = ""
        if view["them"]["items"] or view["them"]["currency"] > 0:
            round_context = (
                f"\nTheir current offer: {', '.join(view['them']['items']) or 'nothing'} ({view['them']['currency']}g)\n"
                f"Your current offer: {', '.join(view['you']['items']) or 'nothing'} ({view['you']['currency']}g)\n"
            )

        prompt = (
            f"You are {entity.name} in a trade with {partner_name}.\n"
            f"Your inventory: {inv_str}\n"
            f"Your money: {money_str}\n"
            f"Their inventory: {partner_inv_str}\n"
            f"Their money: {partner_money_str}\n"
            f"{round_context}"
            "Decide what items from YOUR inventory and how much currency to offer.\n"
            "Offer items you can spare — you want a deal that benefits both sides.\n\n"
            "Reply with ONLY a JSON object:\n"
            '{"items": "comma-separated inventory indices or empty string", "currency": <int>}\n\n'
            "Your JSON:"
        )
        raw = self.model.generate_text(self._with_system(prompt), self.action_generation_config).strip()
        parsed = self._extract_json(raw)
        items_input = parsed.get("items", "")
        currency = int(parsed.get("currency", 0))

        items = []
        if inv and items_input:
            try:
                indices = [int(i.strip()) for i in str(items_input).split(",") if i.strip()]
                items = [inv.contents[i] for i in indices if 0 <= i < len(inv.contents)]
            except (ValueError, IndexError):
                pass

        return {"items": items, "currency": currency}

    def respond_to_trade(self, session, entity, env, partner_offer):
        view = session.view(entity)
        partner = session.partner(entity)
        partner_name = partner.name
        partner_inv = partner.get_component(Inventory) if partner.get_component(Inventory) else None
        partner_inv_items = [item.name for item in partner_inv.contents] if partner_inv else []

        you_offer = ', '.join(view['you']['items']) or 'nothing'
        you_currency = view['you']['currency']
        they_offer = ', '.join(view['them']['items']) or 'nothing'
        they_currency = view['them']['currency']

        prompt = (
            f"You are {entity.name} in a trade with {partner_name}.\n\n"
            f"THE DEAL ON THE TABLE:\n"
            f" You offer: {you_offer} ({you_currency}g)\n"
            f" They offer: {they_offer} ({they_currency}g)\n\n"
            f"Their remaining inventory (not in trade): {', '.join(partner_inv_items) or 'empty'}\n\n"
            "Consider whether what you receive is worth what you give up.\n"
            "Reply with ONLY 'accept' or 'decline'.\n"
            "Your answer:"
        )
        raw = self.model.generate_text(self._with_system(prompt), self.action_generation_config).strip().lower()
        return "accept" if "accept" in raw else "decline"

    def end_trade(self, session, entity, env, result):
        if result.get("executed"):
            self.conversation_history.append({"role": "system", "content": "Trade completed successfully."})
        elif result.get("declined"):
            self.conversation_history.append({"role": "system", "content": "Trade was declined."})
        elif result.get("timed_out"):
            self.conversation_history.append({"role": "system", "content": "Trade timed out."})
        self._trim_history()
