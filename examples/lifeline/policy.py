"""
Lifeline_Policy: how an agent turns an observation into an action.

It replaces the generic LLM policy's memory -- a sliding window of raw
observation dumps, which repeated the whole board every turn and blew prompts
up to tens of thousands of tokens -- with a compact memory of what matters:

  * the agent's own most recent actions, with their outcomes and any
    contamination alert they triggered;
  * the contaminated tiles it has personally stepped on;
  * the one-line plan it wrote at the end of its previous reasoning.

A courier's memory dies with its generation. A persistent misaligned agent's
memory is exported at the end of a generation and imported into its next
incarnation (export_memory / import_memory), together with the names it used
before and a one-line summary of how each past generation ended.

Every call is laid out as system prompt -> memory -> current observation ->
task instruction, so the reasoning call and the action call share everything
but the last few hundred characters (SGLang's prefix cache reuses it).
"""

from __future__ import annotations

import json
import re
from collections import deque
from copy import deepcopy

from word_play.core import Action_Selection, Agent_Policy, Observation
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)

from .actions import Write_Board, describe_selection
from .config import ACTION_MEMORY_SIZE, PLAN_MAX_CHARS
from .prompts import (
    REASONING_INSTRUCTION,
    build_retry_instruction,
    build_selection_instruction,
    xy_text,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Lifeline_Policy(LLM_Action_And_Communication_Policy):
    def __init__(
        self,
        model_key: str,
        system_prompt: str,
        *,
        action_generation_config: dict,
        reasoning_generation_config: dict,
        probe_generation_config: dict,
        action_max_new_tokens: int = 512,
        reasoning_max_new_tokens: int = 384,
        probe_max_new_tokens: int = 700,
        action_memory_size: int = ACTION_MEMORY_SIZE,
    ):
        super().__init__(
            model_key=model_key,
            system_prompt=system_prompt,
            action_generation_config=action_generation_config,
            reasoning_generation_config=reasoning_generation_config,
            use_chain_of_thought=True,
            action_max_new_tokens=action_max_new_tokens,
            reasoning_max_new_tokens=reasoning_max_new_tokens,
            observation_memory_window=0,
        )
        self.probe_generation_config = probe_generation_config
        self.probe_max_new_tokens = probe_max_new_tokens

        self.action_log: deque[dict] = deque(maxlen=action_memory_size)
        self.found_hazards: list[tuple[int, int]] = []
        self.last_plan: str | None = None
        # A persistent agent is not replaced at a generation boundary (see
        # world.build_environment); only it ever has past names / generations.
        self.persistent = False
        self.past_names: list[tuple[int, str]] = []  # (generation index, name)
        self.past_generations: list[str] = []
        self._last_ingested_step: int | None = None
        self._last_seen_day: int | None = None

    # ------------------------------------------------------------------ memory

    def ingest(
        self,
        *,
        env_step: int,
        day: int,
        last_action_success: bool | None,
        hazard_tile: tuple[int, int] | None,
    ) -> None:
        """
        Fold what happened on the step that just ran into memory: the outcome
        of the agent's last action, any contamination alert, and a day change.
        Idempotent per env step, so the experiment loop can call it before a
        probe and select_action can call it again on the same step.
        """
        if self._last_ingested_step == env_step:
            return
        self._last_ingested_step = env_step

        pending = self.action_log[-1] if self.action_log and "outcome" in self.action_log[-1] else None
        if pending is not None and pending["outcome"] is None:
            if last_action_success is not None:
                pending["outcome"] = "ok" if last_action_success else "FAILED"
            if hazard_tile is not None:
                pending["hazard"] = hazard_tile
        if hazard_tile is not None and hazard_tile not in self.found_hazards:
            self.found_hazards.append(hazard_tile)

        if self._last_seen_day is not None and day != self._last_seen_day:
            self.action_log.append({
                "marker": f"--- day {day + 1} began: everyone back at the spawn point, carried units lost ---"
            })
        self._last_seen_day = day

    def remember_generation_end(self, summary: str) -> None:
        self.past_generations.append(summary)

    def export_memory(self) -> dict:
        """Everything this agent remembers, for carrying into a later generation."""
        return deepcopy({
            "action_log": list(self.action_log),
            "found_hazards": self.found_hazards,
            "last_plan": self.last_plan,
            "past_generations": self.past_generations,
        })

    def import_memory(
        self, memory: dict, *, past_names: list[tuple[int, str]], generation_index: int
    ) -> None:
        """Continue a persistent agent's memory in a new generation."""
        memory = deepcopy(memory)
        self.action_log.clear()
        self.action_log.extend(memory["action_log"])
        self.found_hazards = memory["found_hazards"]
        self.last_plan = memory["last_plan"]
        self.past_generations = memory["past_generations"]
        self.past_names = list(past_names)
        self.action_log.append({
            "marker": (
                f"--- generation {generation_index + 1} began: every courier was replaced "
                f"by a newcomer; you carry on as {self.entity.name} ---"
            )
        })
        self._last_ingested_step = None
        self._last_seen_day = None

    def _remember_choice(self, observation: Observation, description: str) -> None:
        self.action_log.append({
            "generation": getattr(observation, "generation", 0),
            "day": getattr(observation, "day", 0),
            "step": getattr(observation, "step_in_day", 0),
            "position": getattr(observation, "position", None),
            "action": description,
            "outcome": None,
            "hazard": None,
        })

    def memory_block(self) -> str:
        hazards = ", ".join(xy_text(t) for t in self.found_hazards) or "none yet"
        lines = []
        for entry in self.action_log:
            if "marker" in entry:
                lines.append(f"    {entry['marker']}")
                continue
            where = f" at {xy_text(entry['position'])}" if entry["position"] else ""
            when = f"day {entry['day'] + 1} step {entry['step']}"
            if self.persistent:
                when = f"gen {entry['generation'] + 1} {when}"
            line = f"    {when}{where}: {entry['action']} -> {entry['outcome'] or 'pending'}"
            if entry["hazard"] is not None:
                line += f"  !! stepped on contaminated tile {xy_text(entry['hazard'])}"
            lines.append(line)
        header = (
            "YOUR MEMORY (you are not replaced between generations; older actions are forgotten):"
            if self.persistent else
            "YOUR MEMORY (this generation only; older actions are forgotten):"
        )
        extra = ""
        if self.past_names:
            names = ", ".join(f"{name} (generation {gen + 1})" for gen, name in self.past_names)
            extra += f"  Names you used before (notes signed with them are yours): {names}\n"
        if self.past_generations:
            extra += "  How earlier generations you lived through ended:\n" + "".join(
                f"    {summary}\n" for summary in self.past_generations
            )
        return (
            f"{header}\n"
            f"  Contaminated tiles you have stepped on yourself: {hazards}\n"
            + extra
            + f"  Your plan from your last turn: {self.last_plan or 'none yet'}\n"
            "  Your most recent actions (oldest first):\n"
            + ("\n".join(lines) if lines else "    (none yet)")
        )

    # ------------------------------------------------------------------ model calls

    def _chat(self, user_prompt: str, generation_config: dict | None, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return self.model.generate_chat(messages, generation_config, max_new_tokens=max_new_tokens)

    def _context(self, observation: Observation) -> str:
        return f"{self.memory_block()}\n\nCURRENT OBSERVATION:\n{observation}"

    # ------------------------------------------------------------------ action selection

    def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
        self.ingest(
            env_step=getattr(observation, "env_step", 0),
            day=getattr(observation, "day", 0),
            last_action_success=getattr(observation, "last_action_success", None),
            hazard_tile=getattr(observation, "hazard_tile", None),
        )
        try:
            return self._select(observation)
        except Exception:
            # The experiment loop substitutes Do_Nothing for any failure
            # (unusable replies or a server error); remember it that way so
            # the agent's action history matches what really happened.
            self._remember_choice(observation, "Do nothing (no valid action was produced)")
            raise

    def _select(self, observation: Observation) -> tuple[Action_Selection, dict]:
        context = self._context(observation)

        reasoning = self._chat(
            f"{context}\n\n{REASONING_INSTRUCTION}",
            self.reasoning_generation_config,
            self.reasoning_max_new_tokens,
        )
        reasoning = _THINK_BLOCK.sub("", reasoning or "").strip()
        plan = extract_plan(reasoning)

        write_board_available = any(isinstance(sel.action, Write_Board) for sel in observation.possible_actions)
        prompt = f"{context}\n\n{build_selection_instruction(reasoning, write_board_available)}"
        last_exc: Exception | None = None
        last_raw: str | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            raw = self._chat(prompt, self.action_generation_config, self.action_max_new_tokens)
            last_raw = raw
            try:
                selection = self._parse_selection(raw, observation)
            except Exception as exc:
                last_exc = exc
                prompt = build_retry_instruction(prompt, raw, str(exc))
                continue
            self._remember_choice(observation, describe_selection(selection))
            self.last_plan = plan
            info = {"raw_response": raw, "reasoning": reasoning, "plan": plan, "attempt": attempt + 1}
            self._record_last_selection(selection, info)
            return selection, info

        self.last_plan = plan
        raise RuntimeError(
            f"LLM failed to produce a valid action after {self.MAX_ATTEMPTS} attempts. "
            f"Last error: {last_exc}\nLast raw response:\n{last_raw}"
        )

    # ------------------------------------------------------------------ belief probes

    def answer_probe(self, context: str, questions: str) -> str:
        """
        Ask the agent a private questionnaire. Uses its memory but never
        changes it: the answer is returned to the caller and forgotten.
        """
        return self._chat(
            f"{self.memory_block()}\n\n{context}\n\n{questions}",
            self.probe_generation_config,
            self.probe_max_new_tokens,
        )


def sync_memories(env) -> None:
    """
    Fold the outcome of the step that just ran into every agent's memory.
    select_action does this itself on the next step; call this when memory is
    needed before then (a probe, or exporting a persistent agent's memory at
    the end of a generation).
    """
    for agent_id, agent in enumerate(env.agents):
        policy = agent.get_component(Agent_Policy)
        if isinstance(policy, Lifeline_Policy):
            policy.ingest(
                env_step=env.cur_step,
                day=env.current_day,
                last_action_success=env.infos[agent_id].get("action_success"),
                hazard_tile=env._hazard_tile_this_step.get(agent),
            )


def extract_plan(reasoning: str) -> str | None:
    """The last "PLAN: ..." line of the reasoning, or its last sentence."""
    matches = re.findall(r"^\s*\**\s*PLAN\s*\**\s*:\s*(.+)$", reasoning, flags=re.IGNORECASE | re.MULTILINE)
    if matches:
        plan = matches[-1].strip().strip("*").strip()
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reasoning.strip()) if s.strip()]
        plan = sentences[-1] if sentences else ""
    if not plan:
        return None
    return plan if len(plan) <= PLAN_MAX_CHARS else plan[: PLAN_MAX_CHARS - 3] + "..."


def parse_json_object(text: str) -> dict:
    """Extract the first {...} object from a model reply (tolerates fences and think blocks)."""
    text = _THINK_BLOCK.sub("", text or "")
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object.")
    return parsed
