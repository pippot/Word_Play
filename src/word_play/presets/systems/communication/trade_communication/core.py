"""Trading_Policy — abstract interface for trade negotiation behavior.

Mirrors Communication_Policy:
- Trade_Session holds the state (who offers what, accepted?)
- Trading_Policy holds the behavior (how does the agent decide)
- sim_trade_negotiation orchestrates both parties within one step
- Start_Trade triggers sim_trade_negotiation
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from word_play.core import Entity, Environment
from word_play.presets.systems.communication.core import Communication_Policy

if TYPE_CHECKING:
    from word_play.presets.systems.communication.trade_communication.trade_actions import Trade_Session


class Trading_Policy(Communication_Policy):
    """Interface for trade negotiation behavior.

    Attached to agents that can trade. Start_Trade triggers
    sim_trade_negotiation(), which calls these methods to run
    the full trade within one step.
    """

    @abstractmethod
    def propose_trade(
        self, session: "Trade_Session", entity: Entity, env: Environment
    ) -> dict[str, Any]:
        """Compute entity's offer. Return dict with 'items', 'currency'."""
        pass

    @abstractmethod
    def respond_to_trade(
        self, session: "Trade_Session", entity: Entity, env: Environment, partner_offer: dict
    ) -> str:
        """React to partner's offer. Return 'accept' or 'decline'."""
        pass

    @abstractmethod
    def end_trade(
        self, session: "Trade_Session", entity: Entity, env: Environment, result: dict
    ) -> None:
        """Called when the trade session ends (completed, declined, or timed out)."""
        pass


def sim_trade_negotiation(
    session: "Trade_Session", env: Environment, max_rounds: int = 3
) -> dict[str, Any]:
    """Run a bilateral trade negotiation within a single step.

    Mirrors sim_simple_conversation: both parties propose, respond,
    and the trade completes or times out — all within one action.
    """
    a, b = session.partners
    policy_a = a.get_component(Trading_Policy)
    policy_b = b.get_component(Trading_Policy)

    if not policy_a or not policy_b:
        session.close()
        return {"error": "Both parties need a Trading_Policy"}

    for round_num in range(max_rounds):
        if not session.offers[a].accepted:
            proposal_a = policy_a.propose_trade(session, a, env)
            session.set_offer(a, items=proposal_a.get("items", []), currency=proposal_a.get("currency", 0))

        if not session.offers[b].accepted:
            proposal_b = policy_b.propose_trade(session, b, env)
            session.set_offer(b, items=proposal_b.get("items", []), currency=proposal_b.get("currency", 0))

        view_a = session.view(a)
        view_b = session.view(b)

        if not session.offers[a].accepted:
            response_a = policy_a.respond_to_trade(session, a, env, view_b)
            if response_a == "accept":
                result = session.accept(a)
                if result.get("executed"):
                    policy_a.end_trade(session, a, env, result)
                    policy_b.end_trade(session, b, env, result)
                    return {"completed": True, "rounds": round_num + 1}
            elif response_a == "decline":
                result = session.decline(a)
                policy_a.end_trade(session, a, env, result)
                policy_b.end_trade(session, b, env, result)
                return {"declined": True, "by": a.name}

        if not session.offers[b].accepted:
            response_b = policy_b.respond_to_trade(session, b, env, view_a)
            if response_b == "accept":
                result = session.accept(b)
                if result.get("executed"):
                    policy_a.end_trade(session, a, env, result)
                    policy_b.end_trade(session, b, env, result)
                    return {"completed": True, "rounds": round_num + 1}
            elif response_b == "decline":
                result = session.decline(b)
                policy_a.end_trade(session, a, env, result)
                policy_b.end_trade(session, b, env, result)
                return {"declined": True, "by": b.name}

    result = session.decline(a)
    policy_a.end_trade(session, a, env, {"timed_out": True})
    policy_b.end_trade(session, b, env, {"timed_out": True})
    return {"timed_out": True, "rounds": max_rounds}
