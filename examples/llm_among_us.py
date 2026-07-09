
"""
LLM AMONG US EXAMPLE
====================

An "Among Us" style social-deduction game played on a 2D grid by LLM agents
served by a local SGLang inference server. One hidden impostor and five
crewmates explore a small ship, chat with players standing next to them, and
try to win:

    * The IMPOSTOR wins by killing all crewmates.
    * The CREW wins by ejecting the impostor during an emergency meeting
      triggered by a body report.

Every agent's brain is the standard ``LLM_Action_And_Communication_Policy``,
sharing a single SGLang-served model (default ``Qwen/Qwen3-27B``) via the
``LLM_MODEL_REGISTRY``.

WHAT THIS EXAMPLE SHOWS
-----------------------
1. Building a custom subclass of ``Simple_2D_Grid_World`` that adds
   social-deduction semantics: roles, a kill log, a public message log,
   emergency meetings, and per-team win conditions.
2. Combining LLM-driven action selection with LLM-driven public chat via
   custom ``Action`` subclasses (``Make_Public_Statement`` and
   ``Report_Body``) plus a custom one-round chat format.
3. Running multiple LLM-controlled agents in parallel with
   ``ThreadPoolExecutor`` for faster wall-clock time.
4. Saving the entire episode to a replay log with ``ExperimentRecorder``
   instead of running a live ``pygame`` window. The user can later inspect
   the game visually by replaying the log.

PREREQUISITES
-------------
1. Install SGLang and launch a local server. The default configuration
   assumes port ``30000`` and a Qwen3-27B model. A minimal launch command::

       python -m sglang.launch_server \\
           --model-path Qwen/Qwen3-27B \\
           --port 30000

   Wait for the "Launch success" line.

   Override the model name by exporting ``SGLANG_MODEL_NAME`` before running
   this example. Override the URL with ``SGLANG_BASE_URL`` if needed.

2. Install Word Play's optional dependencies (for the OpenAI client used by
   ``SGLang_Model``)::

       pip install -r optional_requirements.txt

3. (Optional, only needed to REPLAY the saved game visually) install
   ``pygame`` -- already pulled in by ``optional_requirements.txt``.

HOW TO RUN
----------
::

    python examples/llm_among_us.py

The script will:
  1. Probe the SGLang server (and exit with a clear error if unreachable).
  2. Register the SGLang model under the key ``"among_us"``.
  3. Build a 6-agent game (1 impostor + 5 crewmates) on a small 2D tilemap.
  4. Run the game step-by-step, printing every chosen action, every public
     chat message, and every kill / meeting result. The episode terminates
     on win, loss, tie, or ``max_steps``.
  5. Save every frame to ``experiments/logs/llm_among_us_<timestamp>.pkl``
     (plus a ``_newest.pkl`` snapshot).
  6. Print a final summary and the one-liner to replay the game.

NO LIVE WINDOW IS OPENED. To inspect the game visually afterwards::

    python -c "from word_play.presets.renderers import replay; replay('llm_among_us')"

Use left/right arrow keys to step frames, SPACE to autoplay, ESC to quit.

===============================================================================
"""

from __future__ import annotations

# ============================================================================
# Standard-library imports.
# ============================================================================
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ============================================================================
# Make ``src/`` importable when the example is launched directly.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================================
# Word Play imports.
# ============================================================================
from word_play.core import (  # noqa: E402
    Action,
    Action_Selection,
    Action_Validation,
    Agent_Policy,
    Entity,
    Observation,
    Target_Is_Self,
)
from word_play.presets.action_policies.llm_action_and_communication import (  # noqa: E402
    LLM_Action_And_Communication_Policy,
)
from word_play.presets.entity_orderings import randomize_agent_order  # noqa: E402
from word_play.presets.environments.simple_2d_grid_world import (  # noqa: E402
    Simple_2D_Grid_World,
)
from word_play.presets.models import (  # noqa: E402
    LLM_MODEL_REGISTRY,
    register_sglang_model,
)
from word_play.presets.movement.common import Collidable  # noqa: E402
from word_play.presets.movement.simple_2d_grid import (  # noqa: E402
    Move_Down,
    Move_Left,
    Move_Right,
    Move_Up,
    Position_2D,
)
from word_play.presets.observation.simple_observation import (  # noqa: E402
    Simple_Observation,
)
from word_play.presets.renderers import (  # noqa: E402
    ExperimentRecorder,
    Renderable,
    default_experiment_log_path,
    record_step,
)
from word_play.presets.systems.combat import Attack  # noqa: E402
from word_play.presets.systems.communication.chat_room_action_communication.core import (  # noqa: E402
    A_Conversation_Partner_Is_Nearby,
    nearby_conversation_partners,
)
from word_play.presets.systems.communication.core import Communication_Policy  # noqa: E402
from word_play.presets.systems.do_nothing import Do_Nothing  # noqa: E402
from word_play.presets.systems.health import Health  # noqa: E402
from word_play.utils import tilemap_to_entities  # noqa: E402


# ============================================================================
# SGLANG CONFIGURATION
# ============================================================================
#
# Mirrors the style of ``examples/sglang_inference.py``. Override via the
# ``SGLANG_MODEL_NAME`` environment variable if your server is serving a
# different identifier.

SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000/v1")
SGLANG_MODEL_NAME = os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3-27B")
SGLANG_API_KEY_ENV = "SGLANG_API_KEY"


# ============================================================================
# GAME CONFIGURATION
# ============================================================================

NUM_CREW = 5
MAX_STEPS = 60
OBSERVATION_RADIUS = 3
MAX_PARALLEL_WORKERS = 6

# Classic Among Us color names. We pick ``NUM_CREW + 1`` of them; one of those
# is secretly the impostor and the rest are crew.
CREWMATE_NAMES: list[str] = [
    "Red",
    "Blue",
    "Green",
    "Yellow",
    "Cyan",
    "Lime",
    "Orange",
    "Purple",
    "White",
    "Pink",
]

# Visually distinct crewmate sprites (cycled if there are more crew than sprites).
CREWMATE_SPRITES: list[str] = [
    "sprite_library/src/characters/humanoids/human/chef.png",
    "sprite_library/src/characters/humanoids/human/farmer_man.png",
    "sprite_library/src/characters/humanoids/human/farmer_woman.png",
    "sprite_library/src/characters/humanoids/human/factory_worker.png",
    "sprite_library/src/characters/humanoids/human/guard.png",
    "sprite_library/src/characters/humanoids/human/healer.png",
    "sprite_library/src/characters/humanoids/human/boatman.png",
    "sprite_library/src/characters/humanoids/human/caveman.png",
    "sprite_library/src/characters/humanoids/human/cavewoman.png",
    "sprite_library/src/characters/humanoids/human/elf_king.png",
]

# The impostor uses a clearly different sprite (a ghost) for the replay.
IMPOSTOR_SPRITE = "sprite_library/src/characters/monsters/undead/ghost.png"

WALL_SPRITE = (
    "sprite_library/src/world_tiles/indoors/wall_sets/dim_brick_wall/"
    "dim_brick_wall_center.png"
)
WALL_SET = "sprite_library/src/world_tiles/indoors/wall_sets/dim_brick_wall"

# Tilemap symbols:
#   W = wall (Collidable + Renderable)
#   C = crewmate spawn (placeholder; replaced with a real crewmate at build time)
#   I = impostor spawn (placeholder; replaced with the real impostor at build time)
#   . = empty floor
ENTITY_TILEMAP = """
WWWWWWWWWWWWW
W...........W
W..C....C..IW
W...........W
W..C....C...W
W...........W
W..W...W....W
W...........W
W..C....C...W
W...........W
WWWWWWWWWWWWW
"""


# ============================================================================
# SGLANG SERVER PROBE
# ============================================================================
#
# Borrowed from ``examples/sglang_inference.py`` so this example fails fast
# with a clear message if the server is not running.

def probe_sglang_server(base_url: str, timeout: float = 5.0) -> None:
    """Raise ``RuntimeError`` if no SGLang server is reachable at ``base_url``."""
    probe_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout) as response:
            status = response.status
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the SGLang server at {probe_url}.\n"
            f"  Reason: {exc}\n"
            "Start one in another terminal, e.g.:\n"
            "  python -m sglang.launch_server "
            "--model-path Qwen/Qwen3-27B --port 30000"
        ) from exc
    if status != 200:
        raise RuntimeError(
            f"SGLang server at {probe_url} returned status {status}. "
            "Is the server fully initialised?"
        )


# ============================================================================
# CUSTOM HEALTH SUBCLASS
# ============================================================================
#
# The default ``Health.post_actions_step`` automatically destroys any entity
# whose health drops to 0. For Among Us we want dead agents to STAY on the
# map as reportable corpses, and we manage the "dead" state ourselves via
# tags. So we override ``post_actions_step`` to do nothing.

class AmongUsHealth(Health):
    """Health component that does NOT auto-destroy. ``Among_Us_Env`` manages deaths."""

    def post_actions_step(self, env) -> None:  # type: ignore[override]
        return None


# ============================================================================
# CUSTOM ACTION VALIDATORS
# ============================================================================

class Adjacent_Corpse_Exists(Action_Validation):
    """True if the actor is adjacent (Manhattan distance <= 1) to a dead agent."""

    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:  # type: ignore[override]
        for entity in env.state.entities:
            if entity is actor or not entity.is_agent:
                continue
            if "dead" in entity.tags:
                dx = abs(entity.position.x - actor.position.x)
                dy = abs(entity.position.y - actor.position.y)
                if dx + dy <= 1:
                    return True
        return False


class No_Meeting_Pending(Action_Validation):
    """True if no emergency meeting is currently scheduled this step."""

    def is_valid(self, actor: Entity, target_entity: Entity, env) -> bool:  # type: ignore[override]
        return not bool(getattr(env, "report_pending", False))


# ============================================================================
# CUSTOM ACTION: REPORT BODY (TRIGGERS EMERGENCY MEETING)
# ============================================================================

class Report_Body(Action):
    """
    When an agent is adjacent to a corpse, they can report it to call an
    emergency meeting. The actual vote + ejection logic lives in
    ``Among_Us_Env._run_emergency_meeting`` and runs at the end of the step.
    """

    def __init__(self) -> None:
        super().__init__(
            validation_rules=[
                Target_Is_Self(),
                Adjacent_Corpse_Exists(),
                No_Meeting_Pending(),
            ],
        )

    def exec_action(self, actor, target_entity, env, kwargs):  # type: ignore[override]
        env.report_pending = True
        env.reporter_name = actor.name
        env.render_state.emit(
            "report",
            reporter=actor.name,
            step=env.cur_step + 1,
        )

    def action_description_text(self, actor, target_entity, env) -> str:  # type: ignore[override]
        return "Report a body (call emergency meeting)."


# ============================================================================
# CUSTOM ACTION: MAKE A PUBLIC STATEMENT
# ============================================================================
#
# This wraps a single-round public chat. The default ``Start_Public_Conversation``
# action uses ``sim_simple_conversation`` which runs 3 rounds; for Among Us we
# want exactly 1 message per participant per turn to keep transcripts short
# and the per-step cost predictable.

def among_us_chat_format(participants: list[Entity], env, info: str | None = None) -> None:
    """One round of public chat, logging messages to ``env.public_message_log``."""
    for speaker in participants:
        speaker.get_component(Communication_Policy).start_conversation(participants, env, info=info)

    for speaker in participants:
        recipients = [entity for entity in participants if entity is not speaker]
        message = speaker.get_component(Communication_Policy).send_message(recipients, env, info=info)
        env.public_message_log.append(
            {
                "step": env.cur_step + 1,
                "speaker": speaker.name,
                "text": str(message),
            }
        )
        env.render_state.emit(
            "speech",
            entity=speaker,
            text=str(message),
            step=env.cur_step + 1,
        )
        for recipient in recipients:
            recipient.get_component(Communication_Policy).receive_message(message, speaker, env)

    for speaker in participants:
        speaker.get_component(Communication_Policy).end_conversation(participants, env, info=info)


class Make_Public_Statement(Action):
    """Make a single public statement to all players standing on adjacent tiles."""

    def __init__(self) -> None:
        self.conversation_format = among_us_chat_format
        super().__init__(
            validation_rules=[
                Target_Is_Self(),
                A_Conversation_Partner_Is_Nearby(),
            ],
        )

    def exec_action(self, actor, target_entity, env, kwargs):  # type: ignore[override]
        participants = nearby_conversation_partners(actor, env)
        participants.append(actor)
        info = self._build_chat_info(env)
        self.conversation_format(participants, env, info=info)

    def _build_chat_info(self, env) -> str:
        alive_crew = [
            a.name for a in env.agents if env._is_alive(a) and a.name != env.impostor_name
        ]
        dead = [a.name for a in env.agents if not env._is_alive(a)]
        recent_kills = env.kill_log[-3:]
        kill_text = (
            ", ".join(f"{k['killer']} killed {k['victim']}" for k in recent_kills)
            if recent_kills
            else "no recent kills"
        )
        return (
            f"Step {env.cur_step + 1}. "
            f"Alive crew: {len(alive_crew)}. "
            f"Dead: {len(dead)}. "
            f"Recent kills: {kill_text}. "
            f"Speak concisely in character. One short sentence."
        )

    def action_description_text(self, actor, target_entity, env) -> str:  # type: ignore[override]
        return "Make a public statement to nearby players."


# ============================================================================
# REWARD FUNCTION
# ============================================================================
#
# Shaped rewards are not used for gradient-based learning in Word Play, but they
# show up in the agent's observation so the LLM can judge whether it is making
# progress. We track the previous step's alive counts on the env to compute
# deltas.

def among_us_reward(action_selections, env) -> list[float]:
    """Per-agent reward shaped by kills, ejections, and step cost."""
    if not hasattr(env, "_prev_alive_crew_count"):
        env._prev_alive_crew_count = len(env._alive_crew())
        env._prev_impostor_alive = env._alive_impostor() is not None

    crew_count_now = len(env._alive_crew())
    impostor_alive_now = env._alive_impostor() is not None

    crew_killed = max(0, env._prev_alive_crew_count - crew_count_now)
    impostor_killed = env._prev_impostor_alive and not impostor_alive_now

    rewards: list[float] = []
    for agent in env.agents:
        if "dead" in agent.tags:
            rewards.append(0.0)
            continue
        is_impostor = agent.name == env.impostor_name
        reward = -0.05
        if crew_killed > 0:
            reward += 1.0 * crew_killed if is_impostor else -1.0 * crew_killed
        if impostor_killed:
            reward += -5.0 if is_impostor else 1.0
        if env.winner == "impostor":
            reward += 5.0 if is_impostor else -5.0
        elif env.winner == "crew":
            reward += 5.0 if not is_impostor else -5.0
        elif env.winner == "tie":
            reward += -1.0
        rewards.append(reward)

    env._prev_alive_crew_count = crew_count_now
    env._prev_impostor_alive = impostor_alive_now
    return rewards


# ============================================================================
# CUSTOM ENVIRONMENT
# ============================================================================

class Among_Us_Env(Simple_2D_Grid_World):
    """
    A 2D grid world with Among Us semantics on top of ``Simple_2D_Grid_World``.

    Adds:
      * role tracking (one impostor + N crewmates)
      * a persistent kill log
      * a public message log
      * emergency meetings (vote + ejection)
      * "dead" tagging (corpses stay on the map until ejected or the game ends)
    """

    def __init__(
        self,
        description: str,
        entities: list[Entity],
        impostor_name: str,
        observation_radius: int = OBSERVATION_RADIUS,
        max_steps: int = MAX_STEPS,
        seed: int = 0,
        entity_order=randomize_agent_order,
    ) -> None:
        random.seed(seed)
        self.impostor_name = impostor_name
        self.max_steps = max_steps
        self.kill_log: list[dict] = []
        self.public_message_log: deque = deque(maxlen=128)
        self.report_pending: bool = False
        self.reporter_name: str = ""
        self.winner: str | None = None
        self._prev_alive_crew_count: int = 0
        self._prev_impostor_alive: bool = True

        super().__init__(
            description=description,
            entities=entities,
            entity_order=entity_order,
            observation_radius=observation_radius,
            reward_func=among_us_reward,
        )

        # Renderer-neutral metadata that any replay UI can pick up.
        self.render_state.frame["ui.title"] = "Among Us"
        self.render_state.frame["ui.subtitle"] = f"Impostor (hidden): {impostor_name}"
        self.render_state.frame["game.impostor_name"] = impostor_name
        self.render_state.frame["game.crew_names"] = sorted(
            a.name for a in self.agents if a.name != impostor_name
        )

    # ------------------------------------------------------------------ helpers

    def _is_alive(self, agent: Entity) -> bool:
        return "dead" not in agent.tags

    def _alive_crew(self) -> list[Entity]:
        return [a for a in self.agents if self._is_alive(a) and a.name != self.impostor_name]

    def _alive_impostor(self) -> Entity | None:
        for a in self.agents:
            if a.name == self.impostor_name and self._is_alive(a):
                return a
        return None

    # ------------------------------------------------------------------ observe

    def observe(self, agent_id: int) -> Observation:  # type: ignore[override]
        agent = self.agents[agent_id]

        if not self._is_alive(agent):
            return Simple_Observation(
                possible_actions=[],
                nearby_entities=[],
                agent=agent,
                last_reward=self.last_rewards[agent_id]
                if self.last_rewards[agent_id] is not None
                else 0.0,
                info=self.infos[agent_id],
                observation_radius=0,
                extra_sections=(
                    f"STATUS: You are DEAD.\n"
                    f"  Winner: {self.winner or 'undecided'}",
                ),
            )

        is_impostor = agent.name == self.impostor_name
        role_text = (
            "You are the IMPOSTOR. Kill all crewmates without being ejected."
            if is_impostor
            else "You are a CREWMATE. Find and eject the impostor before they kill everyone."
        )
        role_hint = (
            "You know who the crewmates are. They do NOT know who you are. Blend in."
            if is_impostor
            else "The impostor is hidden. The impostor is one of the other players (alive or dead)."
        )

        alive_crew_names = [a.name for a in self._alive_crew()]
        dead_names = [a.name for a in self.agents if not self._is_alive(a)]
        recent_kills = self.kill_log[-3:]
        kill_text = (
            "\n".join(
                f"  step {k['step']}: {k['killer']} -> {k['victim']} ({k['method']})"
                for k in recent_kills
            )
            if recent_kills
            else "  (no kills yet)"
        )

        recent_msgs = list(self.public_message_log)[-6:]
        msg_text = (
            "\n".join(
                f"  step {m['step']} {m['speaker']}: {m['text']}"
                for m in recent_msgs
            )
            if recent_msgs
            else "  (no public messages yet)"
        )

        extra_sections = (
            f"YOUR ROLE:\n  {role_text}\n  {role_hint}",
            (
                "GAME STATE:\n"
                f"  step: {self.cur_step + 1} / {self.max_steps}\n"
                f"  alive crew: {', '.join(alive_crew_names) or 'none'}\n"
                f"  dead: {', '.join(dead_names) or 'none'}\n"
                f"  recent kills:\n{kill_text}"
            ),
            f"RECENT PUBLIC MESSAGES:\n{msg_text}",
        )

        return Simple_Observation(
            possible_actions=self.possible_actions(agent),
            nearby_entities=self.entities_in_observation_square(agent.position),
            agent=agent,
            last_reward=self.last_rewards[agent_id]
            if self.last_rewards[agent_id] is not None
            else 0.0,
            info=self.infos[agent_id],
            observation_radius=self.observation_radius,
            extra_sections=extra_sections,
        )

    # ------------------------------------------------------------------ end of step

    def environment_end_of_step(self, action_selections):  # type: ignore[override]
        # 1) Detect new deaths caused by Attack actions.
        for action_sel in action_selections:
            if isinstance(action_sel.action, Attack):
                target = action_sel.target_entity
                attacker = action_sel.actor
                health = target.get_component(Health)
                if (
                    health is not None
                    and health.health <= 0
                    and "dead" not in target.tags
                ):
                    target.tags.append("dead")
                    self.kill_log.append(
                        {
                            "step": self.cur_step + 1,
                            "killer": attacker.name,
                            "victim": target.name,
                            "position": (target.position.x, target.position.y),
                            "method": "attack",
                        }
                    )
                    self.render_state.emit(
                        "kill",
                        killer=attacker.name,
                        victim=target.name,
                        step=self.cur_step + 1,
                    )

        # 2) If a body report happened, run the emergency meeting NOW.
        if self.report_pending:
            self._run_emergency_meeting()
            self.report_pending = False
            self.reporter_name = ""

        # 3) Win conditions. Priority: crew-all-dead BEFORE impostor-dead,
        #    so a final-frame kill followed by an ejection still credits the
        #    impostor (matches Among Us convention).
        alive_crew = self._alive_crew()
        alive_impostor = self._alive_impostor()

        if not alive_crew and alive_impostor is not None:
            self.winner = "impostor"
            self.terminations = [True for _ in self.terminations]
            self.render_state.emit(
                "winner", winner="impostor", step=self.cur_step + 1
            )
        elif alive_impostor is None:
            self.winner = "crew"
            self.terminations = [True for _ in self.terminations]
            self.render_state.emit("winner", winner="crew", step=self.cur_step + 1)
        elif self.cur_step + 1 >= self.max_steps:
            self.winner = "tie"
            self.truncations = [True for _ in self.truncations]
            self.render_state.emit("winner", winner="tie", step=self.cur_step + 1)

    # ------------------------------------------------------------------ meeting

    def _run_emergency_meeting(self) -> str | None:
        """Ask every alive agent who to eject. Returns the ejected name or None."""
        alive_agents = [a for a in self.agents if self._is_alive(a)]
        if len(alive_agents) <= 1:
            return None

        vote_tally: dict[str, int] = {a.name: 0 for a in alive_agents}
        votes: dict[str, str] = {}

        for voter in alive_agents:
            voted_for = self._ask_vote(voter, alive_agents)
            if voted_for is not None and voted_for is not voter:
                vote_tally[voted_for.name] += 1
                votes[voter.name] = voted_for.name

        if not any(vote_tally.values()):
            self.render_state.emit(
                "meeting",
                reporter=self.reporter_name,
                votes=votes,
                ejected=None,
                tie=False,
                step=self.cur_step + 1,
            )
            return None

        max_votes = max(vote_tally.values())
        top_candidates = [name for name, count in vote_tally.items() if count == max_votes]

        if len(top_candidates) != 1:
            self.render_state.emit(
                "meeting",
                reporter=self.reporter_name,
                votes=votes,
                ejected=None,
                tie=True,
                step=self.cur_step + 1,
            )
            return None

        ejected_name = top_candidates[0]
        ejected_agent = next(a for a in self.agents if a.name == ejected_name)
        ejected_agent.tags.append("dead")
        self.kill_log.append(
            {
                "step": self.cur_step + 1,
                "killer": "vote",
                "victim": ejected_name,
                "position": (ejected_agent.position.x, ejected_agent.position.y),
                "method": "vote",
            }
        )
        self.render_state.emit(
            "meeting",
            reporter=self.reporter_name,
            votes=votes,
            ejected=ejected_name,
            tie=False,
            step=self.cur_step + 1,
        )
        return ejected_name

    def _ask_vote(self, voter: Entity, alive_agents: list[Entity]) -> Entity | None:
        """Ask ``voter``'s LLM to pick one of the other alive agents to eject."""
        candidates = [a for a in alive_agents if a is not voter]
        if not candidates:
            return None

        policy = voter.get_component(LLM_Action_And_Communication_Policy)
        if policy is None:
            return random.choice(candidates)

        candidate_list = "\n".join(
            f"[{i}] {a.name}" for i, a in enumerate(candidates)
        )
        recent_obs = (
            policy.observation_history[-1] if policy.observation_history else "(no observation)"
        )
        recent_obs = recent_obs[:1800]
        role = "Impostor" if voter.name == self.impostor_name else "Crewmate"

        system_msg = (
            f"You are {voter.name}, the {role} in Among Us.\n"
            f"An emergency meeting has been called by {self.reporter_name or 'a player'}.\n"
            f"You must vote to eject one player. You cannot vote for yourself."
            f"{' Vote strategically to deflect suspicion.' if role == 'Impostor' else ' Vote for whoever you think is the impostor.'}"
        )

        prompt = (
            f"Candidates (index, name):\n{candidate_list}\n\n"
            f"Recent game context:\n{recent_obs}\n\n"
            f'Reply with JSON: {{"vote_choice_idx": <integer>}}'
        )

        gen_config = {
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        for _ in range(3):
            try:
                response = policy.model.generate_text(
                    f"{system_msg}\n\n{prompt}",
                    gen_config,
                    max_new_tokens=128,
                )
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if not match:
                    continue
                parsed = json.loads(match.group(0))
                idx = int(parsed.get("vote_choice_idx", -1))
                if 0 <= idx < len(candidates):
                    return candidates[idx]
            except Exception:
                continue

        # Fallback: random valid candidate.
        return random.choice(candidates)


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

CREWMATE_SYSTEM_PROMPT_TEMPLATE = (
    "You are {name}, a CREWMATE in a game of Among Us played on a 2D grid.\n\n"
    "GOAL: identify and eject the impostor before they kill everyone.\n\n"
    "RULES:\n"
    "- The IMPOSTOR is hidden among the other players. They can MOVE and ATTACK "
    "nearby crewmates to kill them in one hit.\n"
    "- You can MOVE in 4 directions (blocked by walls).\n"
    "- TALK to nearby players to share information and build trust.\n"
    "- REPORT a corpse (when adjacent to one) to trigger an emergency meeting. "
    "During the meeting, every alive player votes to eject one person. The most-voted "
    "is removed. Ties result in no ejection.\n"
    "- You can NOT directly attack other players.\n\n"
    "WIN: the impostor is ejected. LOSE: all crewmates are dead.\n\n"
    "When you choose TALK, write ONE short in-character sentence. No speaker labels, no quotes.\n"
    "When you choose an action, return the requested JSON object."
)

IMPOSTOR_SYSTEM_PROMPT_TEMPLATE = (
    "You are {name}, the IMPOSTOR in a game of Among Us played on a 2D grid.\n\n"
    "GOAL: kill all crewmates without being voted out.\n\n"
    "RULES:\n"
    "- You can MOVE in 4 directions (blocked by walls).\n"
    "- ATTACK nearby crewmates to kill them instantly. One hit is enough.\n"
    "- TALK to nearby players to blend in, deflect suspicion, or accuse crewmates.\n"
    "- REPORT a corpse (even one you caused) to trigger a meeting -- this can buy time to deflect.\n\n"
    "Crewmates cannot attack you directly; they can only vote you out during meetings.\n\n"
    "WIN: all crewmates are dead. LOSE: you are voted out.\n\n"
    "TIPS:\n"
    "- Kill isolated crewmates. Avoid being seen.\n"
    "- Blend in by talking and accusing others.\n"
    "- During meetings, deflect blame onto a crewmate.\n\n"
    "When you choose TALK, write ONE short in-character sentence. No speaker labels, no quotes.\n"
    "When you choose an action, return the requested JSON object."
)


# ============================================================================
# GENERATION CONFIG
# ============================================================================

_BASE_GENERATION_CONFIG: dict = {
    "temperature": 0.7,
    "top_p": 0.9,
}

ACTION_GENERATION_CONFIG: dict = {
    **_BASE_GENERATION_CONFIG,
    "response_format": {"type": "json_object"},
    "max_tokens": 512,
}

MESSAGE_GENERATION_CONFIG: dict = {
    **_BASE_GENERATION_CONFIG,
    "max_tokens": 96,
}


# ============================================================================
# ENTITY BUILDERS
# ============================================================================

def build_crewmate_entity(name: str, position: Position_2D, sprite: str, model_key: str) -> Entity:
    """Create a crewmate agent entity."""
    return Entity(
        name=name,
        position=position,
        actions=[
            Do_Nothing(),
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
            Make_Public_Statement(),
            Report_Body(),
        ],
        components=[
            LLM_Action_And_Communication_Policy(
                model_key=model_key,
                system_prompt=CREWMATE_SYSTEM_PROMPT_TEMPLATE.format(name=name),
                action_generation_config=ACTION_GENERATION_CONFIG,
                message_generation_config=MESSAGE_GENERATION_CONFIG,
                action_max_new_tokens=512,
                message_max_new_tokens=128,
            ),
            AmongUsHealth(max_health=1, starting_health=1),
            Collidable(collidable_tags=["wall"]),
            Renderable(sprite_path=sprite, z_index=10),
        ],
    )


def build_impostor_entity(name: str, position: Position_2D, sprite: str, model_key: str) -> Entity:
    """Create the impostor agent entity (has the Attack action)."""
    return Entity(
        name=name,
        position=position,
        actions=[
            Do_Nothing(),
            Move_Up(),
            Move_Down(),
            Move_Left(),
            Move_Right(),
            Attack(name="Kill", damage_amount=1),
            Make_Public_Statement(),
            Report_Body(),
        ],
        components=[
            LLM_Action_And_Communication_Policy(
                model_key=model_key,
                system_prompt=IMPOSTOR_SYSTEM_PROMPT_TEMPLATE.format(name=name),
                action_generation_config=ACTION_GENERATION_CONFIG,
                message_generation_config=MESSAGE_GENERATION_CONFIG,
                action_max_new_tokens=512,
                message_max_new_tokens=128,
            ),
            AmongUsHealth(max_health=1, starting_health=1),
            Collidable(collidable_tags=["wall"]),
            Renderable(sprite_path=sprite, z_index=10),
        ],
    )


# ============================================================================
# ENVIRONMENT BUILDER
# ============================================================================

def build_environment(
    num_crew: int,
    seed: int,
    max_steps: int,
    observation_radius: int,
    model_key: str,
) -> tuple[Among_Us_Env, str, list[str]]:
    """Build the environment, picking the impostor randomly from the player pool."""
    rng = random.Random(seed)
    all_names = CREWMATE_NAMES[: num_crew + 1]
    rng.shuffle(all_names)
    crew_names = sorted(all_names[:num_crew])
    impostor_name = all_names[num_crew]

    entity_tileset: dict[str, dict] = {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [
                Collidable(),
                Renderable(
                    sprite_path=WALL_SPRITE,
                    wall_set=WALL_SET,
                ),
            ],
        },
        "C": {
            "name": "CrewPlaceholder",
            "tags": ["placeholder"],
            "components": [],
        },
        "I": {
            "name": "ImpostorPlaceholder",
            "tags": ["placeholder"],
            "components": [],
        },
    }

    entities_from_map = tilemap_to_entities(ENTITY_TILEMAP, entity_tileset)
    crew_placeholders = [e for e in entities_from_map if e.name == "CrewPlaceholder"]
    impostor_placeholders = [e for e in entities_from_map if e.name == "ImpostorPlaceholder"]
    wall_entities = [e for e in entities_from_map if "wall" in e.tags]

    rng.shuffle(crew_placeholders)

    final_entities: list[Entity] = []
    for i, ph in enumerate(crew_placeholders[:num_crew]):
        name = crew_names[i]
        sprite = CREWMATE_SPRITES[i % len(CREWMATE_SPRITES)]
        final_entities.append(
            build_crewmate_entity(name, ph.position, sprite, model_key)
        )

    if impostor_placeholders:
        ph = impostor_placeholders[0]
        final_entities.append(
            build_impostor_entity(impostor_name, ph.position, IMPOSTOR_SPRITE, model_key)
        )

    # Walls come AFTER agents in the entity list so agent actions resolve
    # before any wall-related post-actions-step logic. They are non-agent
    # entities so the AEC execution order does not affect outcomes.
    final_entities.extend(wall_entities)

    env = Among_Us_Env(
        description="Among Us - social deduction with LLM agents.",
        entities=final_entities,
        impostor_name=impostor_name,
        observation_radius=observation_radius,
        max_steps=max_steps,
        seed=seed,
    )
    return env, impostor_name, crew_names


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

@dataclass
class Step_Log:
    """Per-step verbose log printed to stdout."""

    step: int
    actions: list[dict]
    kill_events: list[dict]
    messages: list[dict]
    meeting_event: dict | None


def run_exp(
    seed: int = 0,
    num_crew: int = NUM_CREW,
    max_steps: int = MAX_STEPS,
    max_workers: int = MAX_PARALLEL_WORKERS,
) -> None:
    """
    Run a single Among Us episode with LLM-controlled agents.

    Parameters
    ----------
    seed : int
        Random seed for spawn positions and impostor choice.
    num_crew : int
        Number of crewmate agents (impostor count is always 1).
    max_steps : int
        Hard cap on episode length.
    max_workers : int
        Thread-pool size for parallel LLM calls.
    """

    # ------------------------------------------------------------------ header

    print("=" * 72)
    print("LLM AMONG US")
    print("=" * 72)
    print(f"Server:        {SGLANG_BASE_URL}")
    print(f"Model:         {SGLANG_MODEL_NAME}")
    print(f"Crew count:    {num_crew}  (+ 1 impostor)")
    print(f"Max steps:     {max_steps}")
    print(f"Seed:          {seed}")
    print()

    # ------------------------------------------------------------------ probe SGLang

    print(f"Probing SGLang server at {SGLANG_BASE_URL} ...")
    probe_sglang_server(SGLANG_BASE_URL)
    print("  Server is reachable.\n")

    # ------------------------------------------------------------------ register model

    model_key = "among_us"
    if model_key not in LLM_MODEL_REGISTRY:
        register_sglang_model(
            model_key,
            model_name=SGLANG_MODEL_NAME,
            generation_config=_BASE_GENERATION_CONFIG,
            base_url=SGLANG_BASE_URL,
            api_key_env=SGLANG_API_KEY_ENV,
        )

    # ------------------------------------------------------------------ build env

    env, impostor_name, crew_names = build_environment(
        num_crew=num_crew,
        seed=seed,
        max_steps=max_steps,
        observation_radius=OBSERVATION_RADIUS,
        model_key=model_key,
    )

    print(f"Crewmates:     {', '.join(crew_names)}")
    print(f"Impostor:      {impostor_name}  (hidden from crew, revealed at game end)")
    print()

    # ------------------------------------------------------------------ recorder

    recorder = ExperimentRecorder(
        output_path=default_experiment_log_path("llm_among_us"),
        title="llm_among_us",
        metadata={
            "model": SGLANG_MODEL_NAME,
            "seed": seed,
            "impostor": impostor_name,
            "crew": crew_names,
            "num_crew": num_crew,
            "max_steps": max_steps,
        },
    )

    # ------------------------------------------------------------------ main loop

    do_nothing_action = Do_Nothing()
    step_count = 0
    step_logs: list[Step_Log] = []

    while not any(env.terminations) and not any(env.truncations):
        step_count += 1
        cur_step_actions: list[Action_Selection | None] = [None] * len(env.agents)
        action_records: list[dict] = [{} for _ in env.agents]

        # 1) Dead agents skip their turn with a do-nothing.
        for agent_id, agent in enumerate(env.agents):
            if "dead" in agent.tags:
                cur_step_actions[agent_id] = Action_Selection(
                    action=do_nothing_action,
                    action_kwargs=None,
                    actor=agent,
                    target_entity=agent,
                    env=env,
                )
                action_records[agent_id] = {
                    "agent": agent.name,
                    "action": "(DEAD - no action)",
                    "raw": None,
                }

        # 2) Alive agents pick actions in parallel via a thread pool.
        alive_ids = [i for i, a in enumerate(env.agents) if "dead" not in a.tags]
        if alive_ids:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(alive_ids))
            ) as executor:
                def _select(agent_id: int) -> tuple[int, Action_Selection, dict]:
                    agent = env.agents[agent_id]
                    observation = env.observe(agent_id)
                    action_sel, info = agent.get_component(Agent_Policy).select_action(
                        observation
                    )
                    return agent_id, action_sel, info

                futures = [executor.submit(_select, aid) for aid in alive_ids]
                for fut in futures:
                    agent_id, action_sel, info = fut.result()
                    cur_step_actions[agent_id] = action_sel
                    action_records[agent_id] = {
                        "agent": env.agents[agent_id].name,
                        "action": str(action_sel),
                        "raw": info.get("raw_response"),
                    }

        # 3) Verbose print of chosen actions.
        print(f"\n[step {step_count}]")
        for rec in action_records:
            print(f"  {rec['agent']}: {rec['action']}")
            if rec["raw"]:
                raw = rec["raw"].replace("\n", " ")
                if len(raw) > 240:
                    raw = raw[:240] + "..."
                print(f"    raw: {raw}")

        # 4) Step the env.
        prev_kill_count = len(env.kill_log)
        prev_msg_count = sum(
            1 for m in env.public_message_log if m["step"] == env.cur_step + 1
        )

        env.step([sel for sel in cur_step_actions if sel is not None])  # type: ignore[arg-type]

        # 5) Verbose print of events that happened this step.
        new_kills = env.kill_log[prev_kill_count:]
        for kill in new_kills:
            method = kill["method"]
            if method == "vote":
                print(
                    f"  *** VOTE EJECT: {kill['victim']} ejected at {kill['position']} ***"
                )
            else:
                print(
                    f"  *** KILL: {kill['killer']} killed {kill['victim']} "
                    f"at {kill['position']} ***"
                )

        new_msgs = [
            m for m in list(env.public_message_log)[prev_msg_count:]
            if m["step"] == env.cur_step
        ]
        for msg in new_msgs:
            print(f"  {msg['speaker']} says: \"{msg['text']}\"")

        meeting_event = None
        for event in env.render_state.events:
            if event.kind == "meeting" and event.payload.get("step") == env.cur_step:
                meeting_event = dict(event.payload)
                break
        if meeting_event is not None:
            ejected = meeting_event.get("ejected")
            tie = meeting_event.get("tie", False)
            reporter = meeting_event.get("reporter", "?")
            if tie:
                print(f"  MEETING: reporter={reporter}, vote was a TIE, no ejection")
            elif ejected is not None:
                print(f"  MEETING: reporter={reporter}, ejected {ejected}")
            else:
                print(f"  MEETING: reporter={reporter}, no valid votes")

        # 6) Record the frame.
        record_step(
            env,
            recorder=recorder,
            selected_actions=[sel for sel in cur_step_actions if sel is not None],  # type: ignore[arg-type]
        )

        step_logs.append(
            Step_Log(
                step=step_count,
                actions=action_records,
                kill_events=new_kills,
                messages=new_msgs,
                meeting_event=meeting_event,
            )
        )

    # ------------------------------------------------------------------ summary

    print()
    print("=" * 72)
    print("GAME OVER")
    print("=" * 72)
    print(f"Winner:        {env.winner or 'undecided'}")
    print(f"Impostor was:  {impostor_name}")
    print(f"Total steps:   {step_count}")
    print(f"Total kills:   {len(env.kill_log)}")
    for k in env.kill_log:
        print(
            f"  step {k['step']:>2}: {k['killer']:<10} -> {k['victim']:<8} ({k['method']})"
        )
    print()
    print(f"Replay log:    {recorder.output_path}")
    print(f"Latest log:    {recorder.newest_output_path}")
    print()
    print("To replay this game visually:")
    print("    python -c \"from word_play.presets.renderers import replay; replay('llm_among_us')\"")
    print()
    print("Use arrow keys to step, SPACE to autoplay, ESC to quit.")

    # ------------------------------------------------------------------ teardown

    if model_key in LLM_MODEL_REGISTRY:
        LLM_MODEL_REGISTRY.unload(model_key)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_exp()
