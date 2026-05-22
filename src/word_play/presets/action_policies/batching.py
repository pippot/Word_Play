from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from word_play.core.components import Agent_Policy, Non_Agent_Policy
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy

if TYPE_CHECKING:
    from word_play.core import Action_Selection, Environment, Observation


Policy_Selection_Callback = Callable[["Environment", Any, int, "Action_Selection", dict], None]


def build_policy_step_actions(
    env: "Environment",
    *,
    batched: bool = True,
    max_workers: int | None = None,
    on_selection: Policy_Selection_Callback | None = None,
) -> list["Action_Selection"]:
    """Build one action per agent, batching LLM requests when possible."""
    selections: list[Action_Selection | None] = [None] * len(env.agents)
    observations: list[Observation | None] = [None] * len(env.agents)
    infos: list[dict | None] = [None] * len(env.agents)
    llm_agent_indices: list[int] = []
    llm_policies: list[LLM_Action_And_Communication_Policy] = []
    llm_observations: list[Observation] = []

    for agent_id, agent in enumerate(env.agents):
        observation = env.observe(agent_id)
        observations[agent_id] = observation

        policy = agent.get_component(Agent_Policy)
        if policy is not None:
            if batched and isinstance(policy, LLM_Action_And_Communication_Policy):
                llm_agent_indices.append(agent_id)
                llm_policies.append(policy)
                llm_observations.append(observation)
                continue

            selection, info = policy.select_action(observation)
            selections[agent_id] = selection
            infos[agent_id] = dict(info or {})
            _record_last_selection(policy, selection, infos[agent_id])
            continue

        non_agent_policy = agent.get_component(Non_Agent_Policy)
        if non_agent_policy is None:
            raise ValueError(f"Agent '{agent.name}' is missing an Agent_Policy or Non_Agent_Policy component.")

        selection = non_agent_policy.select_action(possible_actions=env.possible_actions(agent), env=env)
        selections[agent_id] = selection
        infos[agent_id] = {}

    if llm_policies:
        llm_results = LLM_Action_And_Communication_Policy.select_actions_batched(
            llm_policies,
            llm_observations,
            max_workers=max_workers,
        )
        for agent_id, (selection, info) in zip(llm_agent_indices, llm_results):
            selections[agent_id] = selection
            infos[agent_id] = info

    resolved_selections: list[Action_Selection] = []
    for agent_id, selection in enumerate(selections):
        if selection is None:
            raise RuntimeError(f"No action was selected for agent index {agent_id}.")
        info = infos[agent_id] or {}
        observation = observations[agent_id]
        if observation is None:
            raise RuntimeError(f"No observation was built for agent index {agent_id}.")
        if on_selection is not None:
            on_selection(env, observation, agent_id, selection, info)
        resolved_selections.append(selection)

    return resolved_selections


def _record_last_selection(policy: Agent_Policy, selection: "Action_Selection", info: dict) -> None:
    policy._last_action = selection
    policy._last_info = {**info, "_last_action": selection}
