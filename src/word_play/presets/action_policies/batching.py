from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from word_play.core.components import Agent_Policy, Non_Agent_Policy

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
    """Build one action per agent.

    In batched mode, observations are built first, then LLM policies are
    queried concurrently. The environment is only stepped by the caller after
    all actions are returned.
    """
    observations = [env.observe(agent_id) for agent_id in range(len(env.agents))]
    jobs = [_build_policy_job(env, agent_id, observation) for agent_id, observation in enumerate(observations)]

    if batched:
        worker_count = max_workers or len(jobs)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_select_policy_job, jobs))
    else:
        results = [_select_policy_job(job) for job in jobs]

    selections: list[Action_Selection] = []
    for agent_id, (selection, info) in enumerate(results):
        if on_selection is not None:
            on_selection(env, observations[agent_id], agent_id, selection, info)
        selections.append(selection)

    return selections


def _build_policy_job(env: "Environment", agent_id: int, observation: "Observation") -> tuple:
    agent = env.agents[agent_id]
    policy = agent.get_component(Agent_Policy)
    if policy is not None:
        return policy, observation

    return agent.get_component(Non_Agent_Policy), env.possible_actions(agent), env


def _select_policy_job(job: tuple) -> tuple["Action_Selection", dict]:
    if isinstance(job[0], Agent_Policy):
        return _select_agent_policy(job[0], job[1])

    selection = job[0].select_action(possible_actions=job[1], env=job[2])
    return selection, {}


def _select_agent_policy(policy: Agent_Policy, observation: "Observation") -> tuple["Action_Selection", dict]:
    selection, info = policy.select_action(observation)
    info = dict(info or {})
    _record_last_selection(policy, selection, info)
    return selection, info


def _record_last_selection(policy: Agent_Policy, selection: "Action_Selection", info: dict) -> None:
    policy._last_action = selection
    policy._last_info = {**info, "_last_action": selection}
