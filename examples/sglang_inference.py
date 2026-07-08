"""
SGLANG INFERENCE EXAMPLE
========================

This example shows how to plug an SGLang inference server into Word Play
as the "brain" of an LLM-controlled agent. The agent runs the same
``LLM_Action_And_Communication_Policy`` used by every other LLM example
in this repository; only the model preset changes.

WHAT IS SGLANG
--------------
SGLang (https://github.com/sgl-project/sglang) is a high-performance
serving framework for LLMs. It exposes an OpenAI-compatible HTTP API
that speaks the standard ``/v1/chat/completions`` protocol. That means
any OpenAI-aware client (including the ``openai`` Python package) can
talk to it without any glue code.

The :class:`word_play.presets.models.sglang.SGLang_Model` preset is a
thin wrapper that points the ``openai`` client at your local SGLang
server. Everything else in Word Play stays unchanged.

WHAT THIS EXAMPLE SHOWS
-----------------------
1. How to start (or point at) a local SGLang server.
2. How to register an ``SGLang_Model`` in the global ``LLM_MODEL_REGISTRY``
   so an LLM-controlled agent can resolve it by key.
3. How to build a small 1D goal-seeking environment from scratch.
4. The full ``observe -> reason -> act -> step`` cycle, with a printed
   step-by-step trace of the agent's actions and the SGLang server's
   raw responses.

PREREQUISITES
-------------
1. Install the SGLang inference stack on the machine that will run the
   server. SGLang is GPU-focused; see the official docs for the supported
   hardware and CUDA versions. A typical install is::

       pip install --upgrade sglang

2. Install the Word Play optional dependencies (for the OpenAI client
   used by ``SGLang_Model``)::

       pip install -r optional_requirements.txt

3. Launch an SGLang HTTP server in another terminal. The default
   ``SGLang_Model`` points at ``http://localhost:30000/v1``, which is
   SGLang's default port. Example launch command::

       python -m sglang.launch_server \\
           --model-path Qwen/Qwen2.5-1.5B-Instruct \\
           --port 30000

   Wait until the server prints a "Launch success" line.

4. Set ``SGLANG_MODEL_NAME`` below to the same value you gave to
   ``--model-path`` (or leave the default).

HOW TO RUN
----------
::

    python examples/sglang_inference.py

The script will:
  1. Verify that the SGLang server is reachable (and fail with a clear
     message if it is not).
  2. Register the SGLang model under the key ``"goal_line_sglang"``.
  3. Run the Explorer agent for up to 6 steps and print the trace.

================================================================================
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports used below.
# ``os``         : read the (optional) SGLANG_API_KEY env var and probe env.
# ``sys``        : insert src/ into sys.path so the example is runnable
#                  directly from the repo root.
# ``pathlib``    : build absolute paths to the project root in a portable way.
# ``urllib``     : tiny HTTP probe to confirm the SGLang server is up
#                  before we start spending tokens.
# ---------------------------------------------------------------------------
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Add ``src/`` to ``sys.path`` so the example can be launched with
# ``python examples/sglang_inference.py`` without installing the package.
# This mirrors what the other examples in this folder do.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ===========================================================================
# IMPORTS
# ===========================================================================
#
# The framework is organized into two layers:
#
#   word_play.core        - framework primitives (no extra dependencies).
#   word_play.presets     - reusable building blocks (models, policies,
#                           environments, movement systems, ...).
#
# We import only what we need, grouped by subsystem for readability.
from word_play.core import (  # noqa: E402
    # ``Action_Selection`` is the output of an agent's ``select_action``.
    # It bundles the chosen Action, the actor Entity, the target Entity,
    # the keyword arguments, and a reference to the Environment.
    Action_Selection,
    # ``Agent_Policy`` is the abstract base class for decision-making
    # components. ``entity.get_component(Agent_Policy)`` returns the
    # concrete policy attached to the entity.
    Agent_Policy,
    # ``Entity`` is the fundamental object in a Word Play environment.
    Entity,
    # ``Environment`` is the simulation driver. All environments inherit
    # from it. It defines the step cycle, observation, reward, and
    # entity management.
    Environment,
)

# LLM_Action_And_Communication_Policy is the LLM-backed Agent_Policy
# shipped with Word Play. It holds a *string key* into the model
# registry rather than a model object, so any number of agents can
# share one underlying model with zero duplication.
from word_play.presets.action_policies.llm_action_and_communication import (  # noqa: E402
    LLM_Action_And_Communication_Policy,
)

# ``entity_definition_order`` is the default entity-order function: it
# preserves the order entities were defined in. We import it to make
# the constructor argument explicit even though Simple_1D_Grid_World
# would also default to it.
from word_play.presets.entity_orderings import entity_definition_order  # noqa: E402

# ``Simple_1D_Grid_World`` is a ready-made base for one-dimensional
# grid environments. It provides ``observe()``, the movement system,
# and the reward plumbing; we subclass it to add goal-checking logic.
from word_play.presets.environments.simple_1d_grid_world import (  # noqa: E402
    Simple_1D_Grid_World,
)

# ``TimeLimit`` truncates episodes after a maximum number of steps.
# It guards against the LLM getting stuck in an infinite loop.
from word_play.presets.env_wrappers.time_limit import TimeLimit  # noqa: E402

# Model registry imports.
#   ``LLM_MODEL_REGISTRY`` is the global singleton that stores model
#       specs and lazily constructs the actual model on first use.
#   ``SGLang_Model`` is the preset we just added. It points the
#       ``openai`` Python client at an SGLang server.
#   ``register_model`` is the generic helper that stores a
#       (class, kwargs) spec in the registry. We use the dedicated
#       ``register_sglang_model`` helper further down for a cleaner
#       keyword interface, but ``register_model`` is the lowest-common-
#       denominator entry point that all presets share.
from word_play.presets.models import (  # noqa: E402
    LLM_MODEL_REGISTRY,
    SGLang_Model,
    register_sglang_model,
)

# 1D movement primitives.
#   ``Position_1D`` is a position type holding a single integer ``x``.
#   ``Move_Left`` and ``Move_Right`` are actions that change ``x`` by
#       -1 and +1 respectively.
#   ``Do_Nothing`` is a no-op action that lets the agent pass its turn.
from word_play.presets.movement.simple_1d_grid import (  # noqa: E402
    Position_1D,
    Move_Left,
    Move_Right,
)
from word_play.presets.systems.do_nothing import Do_Nothing  # noqa: E402


# ===========================================================================
# SGLANG SERVER PROBE
# ===========================================================================
#
# Before registering the model and running the episode, we check that the
# SGLang server is actually reachable. Without this probe, the failure
# mode would be a stack trace from inside the OpenAI client when the
# first generation call lands; with it, the user gets a single clear
# message telling them how to start the server.
#
# SGLang exposes a ``/v1/models`` endpoint on the same port as the
# OpenAI-compatible API. A ``200`` response means the server is up and
# ready to accept chat-completions requests.

def probe_sglang_server(base_url: str, timeout: float = 5.0) -> None:
    """
    Raise a clear ``RuntimeError`` if the SGLang server is not reachable
    at ``base_url``.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible base URL of the SGLang server (for example
        ``"http://localhost:30000/v1"``).
    timeout : float
        Maximum number of seconds to wait for a response.

    Raises
    ------
    RuntimeError
        If the server cannot be reached or returns a non-2xx status.
    """
    # The ``/v1/models`` endpoint is the standard SGLang health probe.
    probe_url = base_url.rstrip("/") + "/models"
    try:
        # ``urllib`` is part of the standard library, so we use it
        # instead of pulling in ``requests`` for a single GET.
        with urllib.request.urlopen(probe_url, timeout=timeout) as response:
            status = response.status
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the SGLang server at {probe_url}.\n"
            f"  Reason: {exc}\n"
            "Start one in another terminal, e.g.:\n"
            "  python -m sglang.launch_server "
            "--model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000"
        ) from exc

    # SGLang returns 200 with a JSON list of served models.
    if status != 200:
        raise RuntimeError(
            f"SGLang server at {probe_url} returned status {status}. "
            "Is the server fully initialised?"
        )


# ===========================================================================
# REWARD FUNCTION
# ===========================================================================
#
# A reward function in Word Play receives:
#   - ``action_selections``: the list of Action_Selection objects for
#     this step (one per agent).
#   - ``env``:               a reference to the Environment.
#
# It must return a ``list[float]`` with one reward per agent.
#
# Rewards are not used for gradient-based learning (Word Play is not an
# RL framework). They appear in the observation string so that the LLM
# can judge whether it is making progress.

def goal_line_reward(action_selections: list[Action_Selection], env: Environment) -> list[float]:
    """
    +1 if the agent is on the goal tile, -0.05 otherwise.

    The goal is identified by the ``"goal"`` tag, which keeps the reward
    function agnostic to the goal's actual position.
    """
    # Find the (single) goal entity by tag.
    goal = next(entity for entity in env.state.entities if "goal" in entity.tags)
    # One reward per agent currently in the environment.
    return [
        1.0 if agent.position.x == goal.position.x else -0.05
        for agent in env.agents
    ]


# ===========================================================================
# CUSTOM ENVIRONMENT
# ===========================================================================
#
# We subclass ``Simple_1D_Grid_World`` to add goal-checking logic.
#
# ``Simple_1D_Grid_World`` already provides:
#   - the 1D movement system
#   - ``observe()`` returning a ``Simple_Observation`` with formatted text
#   - ``entities_in_observation_radius()`` for filtering nearby entities
#
# We add:
#   - ``goal_reached()``         - checks whether any agent is on the goal
#   - ``environment_end_of_step`` - terminates the episode as soon as
#     the goal is reached

class Goal_Line_Env(Simple_1D_Grid_World):
    """
    A 1D grid world with a single goal tile.

    The episode ends as soon as any agent occupies the goal's position.
    """

    def __init__(
        self,
        description: str,
        entities: list[Entity],
        observation_radius: int = 0,
        entity_order=entity_definition_order,
    ) -> None:
        # Validate that a goal entity exists before initialising the
        # parent constructor. This makes the error message immediate
        # and specific rather than surfacing as a NoneType later.
        self.goal = next(
            (entity for entity in entities if "goal" in entity.tags),
            None,
        )
        if self.goal is None:
            raise ValueError("Goal_Line_Env requires an entity tagged with 'goal'.")

        super().__init__(
            description=description,
            entities=entities,
            entity_order=entity_order,
            observation_radius=observation_radius,
            reward_func=goal_line_reward,
        )

    def goal_reached(self) -> bool:
        """Return True if any agent is on the goal tile."""
        return any(
            agent.position.x == self.goal.position.x for agent in self.agents
        )

    def environment_end_of_step(self, action_selections: list[Action_Selection]) -> None:
        """
        Hook called by ``Environment.step()`` after every action has
        been resolved. We use it to mark the episode as terminated
        the moment the goal is reached.
        """
        if self.goal_reached():
            self.terminations = [True for _ in self.terminations]


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def run_exp() -> None:
    """
    Build the environment, register the SGLang model, and run the
    1D goal-seeking episode.
    """

    # =========================================================================
    # STEP 1: Configure the SGLang server connection
    # =========================================================================
    #
    # These three constants define the SGLang server we will talk to.
    # All three are designed to match the defaults of
    # ``python -m sglang.launch_server`` so the example works with the
    # simplest possible launch command.
    #
    #   SGLANG_BASE_URL : the OpenAI-compatible base URL. The default
    #                     port for SGLang is 30000 and the chat-
    #                     completions path is ``/v1``.
    #   SGLANG_MODEL_NAME : the model identifier the SGLang server was
    #                     started with (``--model-path`` argument). The
    #                     string must match exactly.
    #   SGLANG_API_KEY_ENV : name of the env var holding the API key.
    #                     If the server was started with
    #                     ``--api-key some-secret``, set this to the
    #                     same name and export the value before running
    #                     the example. Set to ``None`` to never look at
    #                     the environment (the preset will then send
    #                     the placeholder string ``"EMPTY"``).
    SGLANG_BASE_URL = "http://localhost:30000/v1"
    SGLANG_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    SGLANG_API_KEY_ENV = "SGLANG_API_KEY"

    # Friendly failure if the user forgot to launch SGLang. Without
    # this probe the first generation call would fail with a
    # connection error from deep inside the OpenAI client, which is
    # hard to interpret.
    print(f"Probing SGLang server at {SGLANG_BASE_URL} ...")
    probe_sglang_server(SGLANG_BASE_URL)
    print("  Server is reachable.\n")

    # =========================================================================
    # STEP 2: Register the SGLang model in the global registry
    # =========================================================================
    #
    # Every model in Word Play lives behind a string key in the global
    # ``LLM_MODEL_REGISTRY``. Agents reference the key; the registry
    # constructs (and caches) the actual model on first use.
    #
    # The dedicated ``register_sglang_model`` helper is the cleanest
    # entry point: it accepts the same keyword arguments as the
    # ``SGLang_Model`` constructor and stores them as a spec.
    #
    # We use a unique key (``"goal_line_sglang"``) so this example does
    # not collide with other examples that may also register models.
    model_key = "goal_line_sglang"

    # ``generation_config`` values are forwarded to SGLang on every
    # call. Anything accepted by the OpenAI chat-completions API is
    # accepted here. We also add the SGLang-specific ``top_k`` via
    # ``extra_body`` to show how to pass non-OpenAI parameters.
    generation_config: dict = {
        "temperature": 0.0,    # greedy decoding for deterministic runs
        "top_p": 1.0,
        # SGLang-specific sampling knobs can be sent through extra_body.
        # The OpenAI client forwards them untouched to the server.
        "extra_body": {
            "top_k": -1,        # -1 disables top-k filtering
        },
    }

    # Register the spec. This is *not* the moment the model is
    # constructed; construction happens on the first resolve().
    register_sglang_model(
        model_key,
        model_name=SGLANG_MODEL_NAME,
        generation_config=generation_config,
        base_url=SGLANG_BASE_URL,
        api_key_env=SGLANG_API_KEY_ENV,
    )

    # We also want the action-selection LLM call to use JSON output.
    # SGLang supports the OpenAI-style ``response_format={"type": "json_object"}``
    # parameter, which makes the policy's JSON parser much more reliable.
    action_generation_config: dict = {
        **generation_config,
        "response_format": {"type": "json_object"},
        "max_tokens": 512,    # some models emit long preambles
    }

    # =========================================================================
    # STEP 3: Create the agent entity
    # =========================================================================
    #
    # The agent is an ``Entity`` with:
    #   - a position (``Position_1D``) starting at x=0,
    #   - a list of three actions (do nothing, move left, move right),
    #   - a single component: the LLM-backed policy.
    #
    # The policy points at ``model_key`` rather than at the model
    # object itself, so it survives model reloads and is shared across
    # any other agents that use the same key.
    system_prompt: str = (
        "You control Explorer in a tiny one-dimensional world. "
        "You can move LEFT (x decreases by 1), move RIGHT (x increases "
        "by 1), or DO NOTHING. Your objective is to reach the entity "
        "named Goal. Choose the action that moves Explorer toward the "
        "goal each turn. Return only the requested JSON action choice."
    )

    explorer = Entity(
        name="Explorer",
        position=Position_1D(0),  # start at x=0
        actions=[
            Do_Nothing(),
            Move_Left(),
            Move_Right(),
        ],
        components=[
            LLM_Action_And_Communication_Policy(
                model_key=model_key,
                system_prompt=system_prompt,
                # The policy will run ``model.generate_text`` with
                # ``action_generation_config`` for the action-selection
                # call. ``action_max_new_tokens`` is mapped to the
                # OpenAI ``max_tokens`` parameter under the hood.
                action_generation_config=action_generation_config,
                action_max_new_tokens=512,
            ),
        ],
    )

    # The goal is a static entity with no actions and no policy.
    # The ``"goal"`` tag is what ``goal_line_reward`` and
    # ``Goal_Line_Env.goal_reached`` look for.
    goal = Entity(name="Goal", position=Position_1D(3), tags=["goal"])

    # =========================================================================
    # STEP 4: Build the environment
    # =========================================================================
    #
    # We wrap ``Goal_Line_Env`` in a ``TimeLimit`` so the episode ends
    # after at most 6 steps, even if the agent never reaches the goal.
    # This protects against the LLM spinning forever on a tricky task.
    env = TimeLimit(
        Goal_Line_Env(
            description=(
                "One-agent action-only SGLang inference demo: "
                "Explorer must reach Goal in a 1D world."
            ),
            entities=[explorer, goal],
            observation_radius=5,    # agent sees 5 tiles in each direction
        ),
        max_episode_steps=6,
    )

    # =========================================================================
    # STEP 5: Print the experiment header
    # =========================================================================
    print("=" * 72)
    print("SGLANG INFERENCE EXAMPLE")
    print("=" * 72)
    print(f"Server base URL:  {SGLANG_BASE_URL}")
    print(f"Model:            {SGLANG_MODEL_NAME}")
    print(f"API key env:      {SGLANG_API_KEY_ENV or '(none, using placeholder)'}")
    print(f"Agent:            Explorer at x=0")
    print(f"Goal:             Goal at x=3")
    print(f"Max steps:        6")
    print()

    # =========================================================================
    # STEP 6: Main loop -- observe, select, step
    # =========================================================================
    #
    # The Word Play step cycle is:
    #   1. For each agent: env.observe(agent_id) returns an Observation.
    #   2. The agent's policy picks an Action_Selection.
    #   3. env.step([action_selection]) runs the full step.
    #
    # For this one-agent example we collect a single action per step.
    # In a multi-agent setup you would loop over env.agents and gather
    # one Action_Selection per agent.
    action_history: list[dict] = []
    cumulative_reward = 0.0
    explorer_id = env.agent_to_idx[explorer]

    while not any(env.terminations) and not any(env.truncations):
        # --- OBSERVE ---------------------------------------------------------
        observation = env.observe(explorer_id)

        # --- SELECT ACTION ---------------------------------------------------
        # The LLM policy does the heavy lifting:
        #   a. Build the prompt: system + memory + observation + action list.
        #   b. Call Model.generate_text() (which is the SGLang chat call).
        #   c. Parse the JSON response to extract action_choice_idx.
        #   d. Validate the selection; retry up to 3 times on parse failure.
        #   e. Return (Action_Selection, info_dict).
        action, info = explorer.get_component(Agent_Policy).select_action(observation)
        position_before = explorer.position.x

        # --- STEP THE ENVIRONMENT -------------------------------------------
        env.step([action])

        # --- LOG -------------------------------------------------------------
        reward = env.last_rewards[explorer_id]
        cumulative_reward += reward
        action_history.append(
            {
                "step": len(action_history),
                "position_before": position_before,
                "action": str(action),
                "position_after": explorer.position.x,
                "reward": reward,
                "raw_response": info.get("raw_response"),
            }
        )

    # =========================================================================
    # STEP 7: Print results
    # =========================================================================
    print("Step-by-step trace:")
    print("-" * 72)
    for row in action_history:
        # Truncate the raw LLM response for readability in the terminal.
        raw = row["raw_response"] or ""
        raw = raw.replace("\n", " ")
        if len(raw) > 200:
            raw = raw[:200] + "..."
        print(
            f"  step={row['step']} "
            f"x:{row['position_before']} -> {row['position_after']} "
            f"action={row['action']} "
            f"reward={row['reward']:.2f}"
        )
        print(f"    raw: {raw}")

    print("\nSummary:")
    print(f"  Final position: {explorer.position.x}")
    print(f"  Goal position:  {goal.position.x}")
    print(f"  Reached goal:   {env.goal_reached()}")
    print(f"  Cumulative reward: {cumulative_reward:.2f}")

    # =========================================================================
    # STEP 8: Optional teardown
    # =========================================================================
    #
    # SGLang inference is a *remote* process from the Python client's
    # point of view: nothing in this script owns the server. We
    # therefore do not stop the server here -- the user is expected to
    # manage it (Ctrl-C in the terminal where it was launched).
    #
    # Unloading the model instance from the registry, on the other
    # hand, is a good idea if you plan to re-register or to free
    # resources used by the OpenAI client (the connection pool, etc.).
    if model_key in LLM_MODEL_REGISTRY:
        LLM_MODEL_REGISTRY.unload(model_key)

    print("\nDone.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_exp()
