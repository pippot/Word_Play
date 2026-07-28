"""
HUGGINGFACE LOCAL MODEL EXAMPLE
================================

This example demonstrates how to use a Hugging Face Transformers model as the
"brain" of an agent in a Word Play environment. Everything runs locally on your
machine — no API keys, no cloud services, no internet required after the model
is downloaded.

WHAT THIS EXAMPLE SHOWS
-----------------------
1. How to register and configure a HuggingFace_Model in the global model registry.
2. How to build a custom environment (GoalLine2D) from scratch.
3. How to create an LLM-controlled agent using LLM_Action_And_Communication_Policy.
4. The full observe -> reason -> act -> step cycle with printed trace output.
5. Chain-of-thought reasoning: the agent "thinks step by step" before choosing
   an action.
6. How to *record* an experiment to a pickle file and *replay* it in a pygame
   viewer afterwards — including how the capture-then-replay pipeline works
   under the hood.

PREREQUISITES
-------------
Install the Hugging Face ecosystem:

    pip install transformers torch
    pip install -r optional_requirements.txt   # for pygame (replay viewer)

This downloads the model and tokenizer on first run (typically 1-4 GB for a
1-3B parameter model). The model is cached on disk afterward.

HOW TO RUN
----------
    python examples/huggingface_local_model.py

After the LLM finishes its episode, a pygame replay viewer opens automatically.

You can control which model is used via the MODEL_NAME variable inside run_exp().
Good small defaults: "Qwen/Qwen3-0.6B" (fast on CPU) or
"google/gemma-2-2b-it" or "HuggingFaceTB/SmolLM2-360M-Instruct".

================================================================================

TABLE OF CONTENTS (read in order if you are new to Word Play)
---------------------------------------------------------------------
  I.   WHAT IS WORD PLAY?                         - framework philosophy
  II.  CORE CONCEPTS                               - the building blocks
  III. LLM AGENTS IN DETAIL                        - how the model talks to the env
  IV.  RECORDING AND REPLAY                        - capture-then-replay pipeline
  V.   CODE WALKTHROUGH                            - annotated line-by-line
  VI.  RUNNING THE EXAMPLE                         - what to expect

================================================================================
I. WHAT IS WORD PLAY?
================================================================================

Word Play is a Python framework for building text-first, multi-agent
environments. "Text-first" means that every observation an agent receives is a
plain string -- readable by both humans and LLM-based policies. There is no
pixel-based rendering required (though pygame support is available).

The design philosophy has three pillars:

  1. Composability via Entity-Component
     Entities (agents, items, walls, goals) are plain Python objects. Behaviours
     like health, inventory, or movement are *components* that you mix into an
     entity. Writing a new behaviour means writing a new Component subclass.

  2. The AEC Protocol
     The Agent-Environment Cycle defines how a step executes:
     a. Observe   - each agent receives a text Observation.
     b. Select    - each agent's policy chooses an Action_Selection.
     c. Pre-step  - environment-wide logic runs.
     d. Execute   - all entities execute actions in a defined order.
     e. Post-step - environment-wide logic runs (e.g., check win/loss).
     f. Reward    - the reward function computes one reward per agent.
     g. Reorder   - entities are reordered for the next step.

     Since actions execute sequentially, conflicts (e.g., two agents reaching
     for the same item) are resolved by execution order. The first agent to act
     gets the item; the second's action fails.

  3. Pluggable Policies
     An agent's decision-making is encapsulated in its *policy* -- a component
     that implements Agent_Policy. Three built-in policies ship with the
     framework:
       - Human_Takes_Action:     a human types action indices into the terminal.
       - Random_Action_Policy:   picks uniformly at random.
       - LLM_Action_And_Communication_Policy: calls an LLM to decide.

     You can write your own policy by subclassing Agent_Policy.

================================================================================
II. CORE CONCEPTS
================================================================================

Entity
------
The fundamental object in an environment. Every Entity has:
  - name:       a human-readable string.
  - position:   a Position object (e.g., Position_1D, Position_2D,
                Single_Point_Position for non-spatial environments).
  - tags:       labels like "item", "wall", "goal" used by other systems.
  - actions:    a list of Action instances the entity can perform.
  - components: a list of Component instances (one per component TYPE).

Components are indexed by their exact Python type. You retrieve one with
entity.get_component(SomeComponent).

Action
------
An Action is a callable object that describes:
  - validation_rules: conditions that must hold for the action to be valid
    (e.g., Target_Is_Self, Target_Not_Self, Target_Is_Nearby).
  - required_kwargs:  optional typed keyword arguments (e.g., Int_Arg,
                      String_Choice_Arg) the action needs.
  - exec_action():    the actual behaviour -- moves the entity, deals damage,
                      spawns objects, etc.

When an action executes, it receives (actor, target_entity, env, kwargs) and
optionally returns a dict with execution info (e.g., dice roll result).

Component
---------
Reusable state and behaviour attached to an Entity. Components have lifecycle
hooks:
  - post_initialization()   - after all components are attached.
  - on_instantiation()       - when the entity enters the environment.
  - pre_actions_step()       - before action execution.
  - post_actions_step()      - after action execution.
  - on_destroy()             - when the entity is removed.

Every component that IS-A Agent_Policy also owns the agent's decision-making
via select_action().

Environment
-----------
The simulation driver. Key responsibilities:
  - Owns the entity list (state.entities).
  - Defines the movement system (how positions relate, what "nearby" means).
  - Provides observe(agent_id) -> Observation.
  - Runs the step cycle (step(action_selections)).
  - Computes rewards via reward_func().
  - Reorders entities each step via entity_order().

Observation
-----------
A dataclass (ABC) whose only required field is possible_actions:
the list of Action_Selection objects the agent can choose from.

The concrete subclass Simple_Observation adds:
  - agent state (name, position, health, inventory, etc.)
  - nearby entities
  - last reward
  - extra_sections (custom text blocks the environment injects)
  - formatted action list with indices

The __str__ method produces the text that both humans and LLMs read.

================================================================================
III. LLM AGENTS IN DETAIL
================================================================================

An LLM-powered agent uses three layers:

  Layer 1:  Model (ABC)
  ---------
  Provider-agnostic interface. The only required method is:
      generate_chat(messages, generation_config, max_new_tokens) -> str

  Messages are sequences of Chat_Message(role, content) dicts with roles
  "system", "user", or "assistant".

  Concrete implementations:
    - HuggingFace_Model:   local transformers pipeline.
    - OpenAI_Model:        OpenAI API.
    - OpenRouter_Model:    OpenRouter or any OpenAI-compatible endpoint.
    - Claude_Model:        Anthropic API.
    - Gemini_Model:        Google Gen AI API.
    - Human_Model:         asks a human to type a response.

  Layer 2:  Model_Registry
  ---------
  A central singleton (LLM_MODEL_REGISTRY) that:
    - Stores model specs (class + kwargs) without constructing them.
    - Constructs the model on the first call to resolve(key).
    - Returns the same cached instance on subsequent calls.

  This means 100 agents can share one model object with zero duplication.

  Layer 3:  LLM_Action_And_Communication_Policy
  ---------
  A combined Agent_Policy + Communication_Policy that:

  1. Builds a prompt from the observation string:
       system_prompt (from constructor) +
       observation memory (recent past observations) +
       conversation memory (recent dialogue) +
       current observation +
       formatted action list  (indices + descriptions)

  2. (Optional) Chain-of-thought:
       First sends a "think step by step" reasoning prompt.
       The reasoning text is then injected into the selection prompt.

  3. Sends the prompt to Model.generate_text().

  4. Parses the JSON response:
       {"action_choice_idx": <int>, "action_kwargs": {<key>: <value>, ...}}

  5. Validates the selection.
     If parsing fails, it retries up to 3 times with an error message.

  6. Records the observation in a rolling memory buffer.

HuggingFace_Model specifics
---------------------------
The HuggingFace_Model wraps transformers.pipeline(task="text-generation").

Key configuration options:
  - model_name:      Hugging Face Hub ID (e.g., "Qwen/Qwen3-0.6B")
  - device:          "cpu", "cuda", "mps", or integer device ID.
  - model_kwargs:    passed to from_pretrained(). Useful settings:
                       {"torch_dtype": "auto", "device_map": "auto"}
  - generation_config: passed to pipeline() call. Useful settings:
                       {"temperature": 0.0, "max_new_tokens": 256}
  - tokenizer_name:  optional separate tokenizer. Defaults to model_name.

Since this runs entirely locally, there are:
  - No API calls.
  - No internet required after the model is cached.
  - No API keys or environment variables.

================================================================================
IV. RECORDING AND REPLAY
================================================================================

Word Play has a built-in capture-then-replay pipeline. This example uses it to
record every step of the LLM's episode and then open a pygame viewer so you can
watch the agent navigate the grid.

The pipeline has three stages:

  Stage 1:  Frame capture (interactive_env.py)
  ---------
  capture_environment_frame(env) snapshots the entire environment state into a
  JSON-safe dictionary:

      {
        "cur_step": int,
        "entities": [{"name", "x", "y", "tags", "renderable": {...}, ...}, ...],
        "render_state_frame": {...},       # renderer hints (camera, sidebar, etc.)
        "render_state_events": [...],      # transient effects (speech bubbles, hits)
        "agent_observations": [{"agent_name", "text", "possible_actions"}, ...],
      }

  Entity references inside render payloads are replaced with {"__entity_ref__": N}
  so they can be resolved during replay.

  The Renderable component on each entity is what gives it a sprite path and
  z-ordering in the viewer. Without Renderable, entities are invisible.

  Stage 2:  ExperimentRecorder (interactive_env.py)
  ---------
  ExperimentRecorder wraps the frame capture:
      recorder.record(env) -> captures frame, appends, and pickle-dumps the
      full payload to both {slug}_{timestamp}.pkl and {slug}_newest.pkl.

  The convenience function record_step(env, title=...) creates or reuses a
  default recorder and calls record() for you. Files go to experiments/logs/.

  Stage 3:  Replay (replay_and_live.py)
  ---------
  replay(log_path, renderer) loads the pickle and opens a pygame window:

      ReplayFrameEnvironment(frame)       - wraps one serialized frame, rebuilds
                                             Entity objects, decodes render events
      replay_frames(renderer, frames)     - the main loop: renders each frame,
                                             handles keyboard navigation

  The ReplayFrameEnvironment._build_entities() reconstructs Entities with
  Position_2D and any serialized components (including Renderable). It shares
  the same render_environment() draw pipeline that live rendering uses.

  Keyboard controls in the replay viewer:
      SPACE        - toggle play/pause autoplay
      RIGHT arrow  - step forward one frame
      LEFT arrow   - step backward one frame
      HOME         - jump to first frame
      END          - jump to last frame
      R            - restart from frame 0
      ESC          - exit

================================================================================
V. CODE WALKTHROUGH
================================================================================

"""

from __future__ import annotations

import os
import sys
from typing import Callable

# ---------------------------------------------------------------------------
# Add src/ to sys.path so the package is importable when running from the
# examples/ directory as `python examples/huggingface_local_model.py`.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ===========================================================================
# IMPORTS
# ===========================================================================
#
# The framework is organized into two layers:
#
#   word_play.core        - framework primitives (no dependencies)
#   word_play.presets     - reusable building blocks
#
# We import only what we need.

from word_play.core import (
    # Action_Selection bundles: the Action, the actor Entity, the target Entity,
    # the keyword arguments, and a reference to the Environment. It is the
    # output of every agent's select_action().
    Action_Selection,

    # Agent_Policy is the abstract base class for all agent decision-making
    # components. Its only abstract method is:
    #     select_action(observation) -> tuple[Action_Selection, dict]
    Agent_Policy,

    # Entity is the fundamental object in a Word Play environment.
    Entity,

    # Environment is the simulation driver. All environments inherit from it.
    # It defines the step cycle, observation, reward, and entity management.
    Environment,

    # Observation is the abstract base class for what an agent "sees" each
    # step. Its only required field is `possible_actions`.
    Observation,
)

# LLM_Action_And_Communication_Policy is a concrete Agent_Policy that uses a
# Model instance to select actions. It also implements Communication_Policy
# for multi-agent message passing, but in this example we only use the action
# selection side.
from word_play.presets.action_policies.llm_action_and_communication import (
    LLM_Action_And_Communication_Policy,
)

# The entity_order function determines the sequence in which entities execute
# their actions each step. entity_definition_order preserves the order they
# were listed when the environment was constructed.
from word_play.presets.entity_orderings import entity_definition_order

# Simple_2D_Grid_World is a ready-made base for two-dimensional grid
# environments. It provides observe(), possible_actions(), the movement system
# (INFINITE_2D_MOVEMENT_SYSTEM), and entities_in_observation_square(). We
# subclass it to add goal-checking logic.
#
# Why 2D instead of 1D? The replay infrastructure
# (ReplayFrameEnvironment._build_entities()) reconstructs every entity with
# Position_2D(x, y). If we used Position_1D (which has no .y attribute), the
# replay viewer would skip those entities entirely. Using 2D gives us visual
# replay at no extra complexity cost.
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World

# TimeLimit is a wrapper that truncates episodes after a maximum number of
# steps (useful for preventing infinite loops in LLM-controlled episodes).
from word_play.presets.env_wrappers.time_limit import TimeLimit

# The model registry is a singleton that stores model specs and lazily
# constructs model instances. Agents hold only a string key into this
# registry.
from word_play.presets.models import (
    LLM_MODEL_REGISTRY,

    # HuggingFace_Model wraps transformers.pipeline("text-generation").
    HuggingFace_Model,

    # register_model is the generic function that stores a (class, kwargs)
    # spec in the registry. register_huggingface_model is a convenience
    # wrapper with keyword-friendly parameters.
    register_model,
)

# Position_2D is a position type for two-dimensional grids (x, y coordinates).
from word_play.presets.movement.simple_2d_grid import Position_2D

# Move_Left and Move_Right are actions that change the entity's x coordinate
# by -1 and +1 respectively. Do_Nothing is a no-op.
from word_play.presets.movement.simple_2d_grid import Move_Left, Move_Right
from word_play.presets.systems.do_nothing import Do_Nothing

# ---------------------------------------------------------------------------
# RECORDING AND REPLAY IMPORTS
# ---------------------------------------------------------------------------
#
# Renderable is a Component that associates an entity with a sprite file path
# and z-ordering. Without Renderable, entities have no visual representation
# in the replay viewer.
#
# Pygame_Renderer is the pygame-based renderer implementation. Grid_Layout_Adapter
# tells it how to map 2D grid coordinates to screen pixels.
#
# record_step is a convenience function that captures the current environment
# state and appends it to an ExperimentRecorder pickle file.
#
# default_experiment_log_path returns the file path where recordings are
# saved (experiments/logs/{title}_{timestamp}.pkl).
#
# replay loads a recording pickle and opens an interactive pygame viewer.

from word_play.presets.renderers import (
    Grid_Layout_Adapter,
    Pygame_Renderer,
    Renderable,
    default_experiment_log_path,
    record_step,
    replay,
)

# ===========================================================================
# REWARD FUNCTION
# ===========================================================================
#
# A reward function receives:
#   - action_selections: the list of Action_Selection objects for this step
#     (one per agent).
#   - env:               a reference to the Environment.
#
# It must return a list[float] with one reward per agent.
#
# Rewards in Word Play are not used for gradient-based learning (Word Play is
# not a RL framework). They serve as a signal in the observation string so
# that an LLM can judge whether it is making progress.

def goal_line_reward(action_selections: list[Action_Selection], env: Environment) -> list[float]:
    """
    Simple reward: +1 if the agent is at the goal, -0.05 otherwise.

    The goal is identified by a tag ("goal"). We find it by scanning the
    environment's entity list.

    This reward is displayed in the observation string each step, allowing
    the LLM to see whether it is getting closer to or farther from the goal.
    """
    goal = next(entity for entity in env.state.entities if "goal" in entity.tags)
    return [1.0 if agent.position.x == goal.position.x else -0.05 for agent in env.agents]

# ===========================================================================
# CUSTOM ENVIRONMENT
# ===========================================================================
#
# We define a custom environment by subclassing Simple_2D_Grid_World.
#
# Simple_2D_Grid_World already provides:
#   - INFINITE_2D_MOVEMENT_SYSTEM (unbounded 2D grid movement)
#   - observe() that returns a Simple_Observation with formatted text
#   - entities_in_observation_square() for filtering nearby entities
#
# We add:
#   - goal_reached():   a helper that checks if any agent is at the goal.
#   - environment_end_of_step(): if the goal is reached, terminate the
#     episode (set self.terminations = [True, ...]).

class GoalLine2D(Simple_2D_Grid_World):
    """
    A two-dimensional grid world where an agent must reach a goal entity.

    The environment terminates as soon as the agent's position matches the
    goal's position on both x and y axes. In this example everything lives on
    y=0, creating a simple 1D corridor within the 2D system -- but using 2D
    positions means the replay viewer can render the entities correctly.

    Parameters
    ----------
    description : str
        A human-readable description passed to the parent Environment.
    entities : list[Entity]
        Must contain exactly one entity tagged with "goal".
    observation_radius : int
        How far the agent can "see" in each direction (Manhattan distance).
        radius=5 means the agent sees 5 tiles in each cardinal direction.
    entity_order : callable
        Function that reorders entities each step.
    """

    def __init__(
        self,
        description: str,
        entities: list[Entity],
        observation_radius: int = 0,
        entity_order: Callable[[list[Entity], Environment], list[int]] = entity_definition_order,
    ) -> None:
        # Validate that a goal entity exists before initialising the parent.
        self.goal = next((entity for entity in entities if "goal" in entity.tags), None)
        if self.goal is None:
            raise ValueError("GoalLine2D requires an entity tagged with 'goal'.")

        super().__init__(
            description=description,
            entities=entities,
            entity_order=entity_order,
            observation_radius=observation_radius,
            reward_func=goal_line_reward,
        )

    def goal_reached(self) -> bool:
        """
        Check whether any agent has reached the goal position.

        Both x and y must match. (In this example y is always 0 for all
        entities, so this is effectively an x-only check, but the 2D
        comparison is general.)
        """
        return any(
            agent.position.x == self.goal.position.x and agent.position.y == self.goal.position.y
            for agent in self.agents
        )

    def environment_end_of_step(self, action_selections: list[Action_Selection]) -> None:
        """
        Called by Environment.step() after all actions have been executed.

        Here we check termination. If the goal is reached, every agent's
        termination flag is set to True, causing the main loop to exit.
        """
        if self.goal_reached():
            self.terminations = [True for _ in self.terminations]

    # NOTE: environment_start_of_step() is abstract in Environment. We do not
    # need any start-of-step logic, so we skip it -- Simple_2D_Grid_World
    # already provides a no-op implementation.

# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def run_exp():
    # =========================================================================
    # STEP 1: Choose and configure the Hugging Face model
    # =========================================================================
    #
    # Every model in Word Play is registered in the global LLM_MODEL_REGISTRY
    # under a string key. Agents reference this key -- they never hold the model
    # object directly. This allows any number of agents to share one model
    # without duplication.
    #
    # Model registration stores only a spec (class + constructor kwargs). The
    # model itself is NOT constructed yet. Construction happens lazily on the
    # first call to LLM_MODEL_REGISTRY.resolve(key). This means the heavy
    # model download / loading only happens when the first agent actually needs
    # to generate text.
    #
    # -------------------------------------------------------------------------
    # MODEL SELECTION GUIDE
    # -------------------------------------------------------------------------
    # The model_name must be a Hugging Face Hub ID. Here are suggestions
    # ordered from smallest/fastest to most capable:
    #
    #   "HuggingFaceTB/SmolLM2-360M-Instruct"   - 360M params, runs on any CPU
    #   "Qwen/Qwen3-0.6B"                      - 0.6B params, good balance
    #   "Qwen/Qwen2.5-1.5B-Instruct"            - 1.5B params, needs ~4 GB RAM
    #   "google/gemma-2-2b-it"                  - 2B params, strong quality
    #   "microsoft/Phi-3-mini-4k-instruct"      - 3.8B params, needs ~8 GB RAM
    #
    # The first download will be large (1-4 GB depending on model). The
    # transformers library caches it in ~/.cache/huggingface/hub/ for reuse.
    # -------------------------------------------------------------------------

    model_key = "hf_local"
    model_name = "Qwen/Qwen3-0.6B"

    # generation_config is passed to the pipeline call (i.e., to
    # model.generate()). These are the text-generation parameters.
    generation_config = {
        "temperature": 0.0,           # 0 = greedy decoding (deterministic)
        "do_sample": False,           # no random sampling
    }

    # The HuggingFace_Model does not need any API key (public models). We pass
    # token_env=None to skip the environment variable check.
    register_model(
        model_key,
        HuggingFace_Model,
        model_name=model_name,
        generation_config=generation_config,
        device="cpu",                               # use "cuda" if you have a GPU
        token_env=None,                             # no API key needed
        model_kwargs={                              # passed to from_pretrained()
            "torch_dtype": "auto",                  # use the model's native dtype
        },
    )

    # The LLM_Action_And_Communication_Policy needs a separate generation
    # config for action selection that forces JSON output mode. (The
    # HuggingFace model will receive an instruction to output JSON; the
    # config just controls temperature / token limits.)
    action_generation_config = {
        **generation_config,
        "max_new_tokens": 512,          # generous limit — some models (Qwen3) emit <think> blocks before the JSON
    }

    # Chain-of-thought config -- a separate generation call for the reasoning
    # step. We use slightly more tokens because reasoning takes more text.
    reasoning_generation_config = {
        **generation_config,
        "max_new_tokens": 256,
    }

    # =========================================================================
    # STEP 2: Create the agent entity with an LLM policy
    # =========================================================================
    #
    # An Entity is defined by:
    #   name       - displayed in observations.
    #   position   - where it is in the world.
    #   actions    - the Action instances it can perform.
    #   components - the Component instances that give it behaviour.
    #
    # The LLM_Action_And_Communication_Policy is the key component: it
    # implements Agent_Policy and connects to the HuggingFace model.
    #
    # We also add a Renderable component so the entity appears in the replay
    # viewer. Renderable stores a sprite file path and a z-index (sorting
    # order for overlapping sprites).

    system_prompt = (
        "You control Explorer in a simple grid world. "
        "The world is a flat grid. You can move LEFT (decrease x by 1), "
        "move RIGHT (increase x by 1), or do NOTHING. "
        "Your goal is to reach the entity named 'Goal'. "
        "Move toward Goal by choosing the appropriate action each turn. "
        "Return only the requested JSON action choice - no extra text, no markdown."
    )

    explorer = Entity(
        name="Explorer",
        position=Position_2D(1, 0),   # start near the left side
        actions=[
            Do_Nothing(),
            Move_Left(),
            Move_Right(),
        ],
        components=[
            LLM_Action_And_Communication_Policy(
                model_key=model_key,
                system_prompt=system_prompt,
                use_chain_of_thought=True,               # enable step-by-step reasoning
                action_generation_config=action_generation_config,
                reasoning_generation_config=reasoning_generation_config,
                action_max_new_tokens=512,
                reasoning_max_new_tokens=512,
            ),
            # Renderable gives this entity a sprite in the replay viewer.
            # The sprite_path is relative to the project root.
            Renderable(
                sprite_path="sprite_library/src/characters/humanoids/human/ordinary_human.png",
                z_index=10,
            ),
        ],
    )

    # The goal is a simple entity tagged with "goal". It has no actions and
    # no LLM policy -- it just sits at a position. We give it a Renderable
    # so it appears as a treasure chest in the replay viewer.
    goal = Entity(
        name="Goal",
        position=Position_2D(4, 0),    # on the same y=0 row, a few steps right
        tags=["goal"],
        components=[
            Renderable(
                sprite_path="sprite_library/src/items/misc/chest.png",
                z_index=5,
            ),
        ],
    )

    # =========================================================================
    # STEP 3: Build the environment
    # =========================================================================
    #
    # We wrap the GoalLine2D in a TimeLimit so the episode ends after a fixed
    # number of steps even if the goal hasn't been reached. This prevents
    # infinite loops in the LLM.
    #
    # We do NOT attach a Pygame_Renderer to the environment here. Recording
    # (capture_environment_frame) saves entity state and render events from
    # env.render_state regardless of whether a renderer is attached. The
    # replay viewer creates its own renderer when replay() is called.

    env = TimeLimit(
        GoalLine2D(
            description="A 2D grid world with an Explorer and a Goal.",
            entities=[explorer, goal],
            observation_radius=5,           # agent sees 5 tiles in each direction
        ),
        max_episode_steps=10,
    )

    # =========================================================================
    # STEP 4: Print the experiment header
    # =========================================================================

    print("=" * 72)
    print("HUGGING FACE LOCAL MODEL EXAMPLE")
    print("=" * 72)
    print(f"Model:     {model_name}")
    print(f"Device:    cpu")
    print(f"Max steps: 10")
    print(f"Agent:     Explorer at (1, 0), Goal at (4, 0)")
    print(f"Chain-of-thought: enabled")
    print(f"Recording to: {default_experiment_log_path('huggingface_goal_seeking')}")
    print()

    # =========================================================================
    # STEP 5: MAIN LOOP -- observe, select, step, record
    # =========================================================================
    #
    # The Word Play step cycle:
    #
    #   1. For each agent, call env.observe(agent_id) to get an Observation.
    #      The observation is a formatted string containing:
    #        - last action result (success/failure + info)
    #        - reward this turn
    #        - the agent's state (name, position, health, inventory)
    #        - visible area description
    #        - nearby entities
    #        - available actions with their indices
    #
    #   2. The agent's policy (Agent_Policy.select_action(observation)) returns
    #      an Action_Selection and an info dict. For LLM policies, the info
    #      dict contains the raw LLM response and any chain-of-thought text.
    #
    #   3. env.step([action_selection]) executes one step of the simulation:
    #      a. environment_start_of_step()      - custom pre-step logic
    #      b. entity.pre_actions_step()         - component hooks (e.g., poison ticking)
    #      c. Action execution (in entity order)
    #         - agents execute their chosen action
    #         - non-agent entities use Non_Agent_Policy to decide
    #      d. entity.post_actions_step()        - component hooks (e.g., death checks)
    #      e. environment_end_of_step()         - custom post-step logic (termination)
    #      f. reward_func()                     - compute rewards for each agent
    #      g. entity_order()                    - reorder entities for next step
    #
    #   4. record_step(env) captures a replay frame and writes it to disk.
    #
    # In our GoalLine2D environment, environment_end_of_step checks whether
    # the agent reached the goal and sets termination flags if so.

    action_history = []
    cumulative_reward = 0.0
    explorer_id = env.agent_to_idx[explorer]

    # env.terminations and env.truncations are lists of booleans, one per
    # agent. The loop continues while no agent has terminated or been truncated.
    while not any(env.terminations) and not any(env.truncations):
        # --- 1. OBSERVE ---
        # env.observe(agent_id) returns a Simple_Observation whose __str__()
        # produces the text prompt that the LLM will read.
        observation = env.observe(explorer_id)

        # --- 2. SELECT ACTION ---
        # explorer.get_component(Agent_Policy) returns the
        # LLM_Action_And_Communication_Policy instance. Its select_action()
        # does the following:
        #   a. If use_chain_of_thought is True:
        #      - Build a "think step by step" reasoning prompt.
        #      - Call Model.generate_text() with the reasoning config.
        #      - Store the reasoning text.
        #   b. Build the action selection prompt:
        #      system_prompt + past observations + conversation history +
        #      current observation + formatted action list.
        #      If chain-of-thought is enabled, the reasoning text is injected
        #      into this prompt.
        #   c. Call Model.generate_text() with the action config.
        #   d. Parse the JSON response to extract action_choice_idx and
        #      optional action_kwargs.
        #   e. If parsing fails, retry up to MAX_ATTEMPTS (default 3) with
        #      an error message telling the model what went wrong.
        #   f. Validate the selected action (constraints still apply).
        #   g. Return tuple[Action_Selection, info_dict].
        #
        # The info_dict contains:
        #   - raw_response: the raw text the LLM returned.
        #   - reasoning:    the chain-of-thought text (if enabled).
        #   - attempt:      how many retries were needed.
        action, info = explorer.get_component(Agent_Policy).select_action(observation)
        position_before = explorer.position.x

        # --- 3. STEP THE ENVIRONMENT ---
        env.step([action])

        # --- 4. RECORD A REPLAY FRAME ---
        # record_step() calls capture_environment_frame(env) which:
        #   - Serializes every entity's name, position (x, y), tags, components
        #     (including Renderable), and whether it is an agent.
        #   - Serializes env.render_state.frame (persistent render hints) and
        #     env.render_state.events (transient per-step effects).
        #   - Serializes the observation text and possible actions for each
        #     agent.
        #   - Replaces Entity object references with {"__entity_ref__": N}
        #     so they can be resolved during replay.
        #
        # The frame is appended to an ExperimentRecorder and the full payload
        # is pickle-dumped to experiments/logs/huggingface_goal_seeking_*.pkl.
        record_step(env, title="huggingface_goal_seeking")

        reward = env.last_rewards[explorer_id]
        cumulative_reward += reward
        action_history.append({
            "step": len(action_history),
            "position_before": position_before,
            "action": str(action),
            "position_after": explorer.position.x,
            "reward": reward,
            "reasoning": info.get("reasoning"),
            "raw_response": info.get("raw_response"),
        })

    # =========================================================================
    # STEP 6: Print results
    # =========================================================================

    print("\nStep-by-step trace:")
    print("-" * 72)
    for row in action_history:
        print(f"  Step {row['step']}:  x={row['position_before']} -> x={row['position_after']}")
        print(f"    Action: {row['action']}")
        print(f"    Reward: {row['reward']}")
        if row["reasoning"]:
            # Truncate long reasoning for clean display
            reasoning = row["reasoning"].replace("\n", "\n    ")
            if len(reasoning) > 400:
                reasoning = reasoning[:400] + "... [truncated]"
            print(f"    Reasoning:\n    {reasoning}")
        if row["raw_response"]:
            raw = row["raw_response"].replace("\n", " ")
            if len(raw) > 200:
                raw = raw[:200] + "..."
            print(f"    Raw:    {raw}")
        print()

    print("Summary:")
    print(f"  Final position:  ({explorer.position.x}, {explorer.position.y})")
    print(f"  Goal position:   ({goal.position.x}, {goal.position.y})")
    print(f"  Goal reached:    {env.goal_reached()}")
    print(f"  Cumulative reward: {cumulative_reward:.2f}")

    # =========================================================================
    # ADDITIONAL NOTE: WHAT JUST HAPPENED?
    # =========================================================================
    #
    # If the agent reached the goal, congratulations -- the local LLM was able
    # to understand the task, parse the observation, and select appropriate
    # movement actions. This is non-trivial! The model had to:
    #
    #   1. Parse the observation text to find its current position.
    #   2. Parse the observation to find the Goal's position.
    #   3. Compare the two to determine direction.
    #   4. Map "move toward" to either Move_Left or Move_Right.
    #   5. Output valid JSON with the correct action_choice_idx.
    #
    # If the agent did NOT reach the goal, common reasons:
    #   - The model is too small to reliably follow instructions (try a larger
    #     model like Qwen2.5-1.5B or Phi-3-mini).
    #   - The model outputs markdown-wrapped JSON or extra text that the parser
    #     rejects. The retry mechanism should handle this, but small models
    #     may keep failing.
    #   - The model "hallucinates" a position and acts on false information.
    #
    # Try adjusting the system_prompt, temperature, or model size to improve
    # results.

    print("\nYour episode has been recorded to:")
    print(f"  {default_experiment_log_path('huggingface_goal_seeking')}")

    # =========================================================================
    # STEP 7: REPLAY THE RECORDED EPISODE
    # =========================================================================
    #
    # Now we open an interactive pygame viewer to watch the recorded episode.
    #
    # How replay() works:
    #
    #   1. It resolves the log path. If the path ends in .pkl it uses it
    #      directly; otherwise it looks for {title}_newest.pkl in the
    #      experiments/logs/ directory.
    #
    #   2. It loads the pickle via load_recording_payload().
    #
    #   3. It creates a Pygame_Renderer (or reuses one we pass in). The
    #      renderer has a Grid_Layout_Adapter that maps grid cells to screen
    #      pixels and draws a floor tile as the background.
    #
    #   4. It calls replay_frames(renderer, frames), which:
    #      a. For each frame, creates a ReplayFrameEnvironment that rebuilds
    #         the Entity objects from serialized data (using the Renderable
    #         component info to determine sprites and z-ordering).
    #      b. Calls render_environment(renderer, replay_env), the same draw
    #         pipeline used for live rendering.
    #      c. Handles keyboard input: SPACE toggles autoplay, arrows step
    #         through frames, ESC exits.
    #
    # The Pygame_Renderer we create here uses:
    #   - Grid_Layout_Adapter: maps grid (x, y) to screen pixels.
    #   - tile_size=56: each grid cell is 56x56 pixels.
    #   - default_floor_sprite: the background tile drawn under entities.

    print("\nOpening replay viewer...")
    print("  SPACE: play/pause    LEFT/RIGHT: step    HOME/END: jump    ESC: exit")

    renderer = Pygame_Renderer(
        Grid_Layout_Adapter(),
        tile_size=56,
        default_floor_sprite="sprite_library/src/world_tiles/indoors/floors/day_grass_floor_c.png",
    )
    replay("huggingface_goal_seeking", renderer)

    print("\nDone.")


if __name__ == "__main__":
    run_exp()

# ===========================================================================
# VI. RUNNING THE EXAMPLE
# ===========================================================================
#
# Terminal:
#     python examples/huggingface_local_model.py
#
# Expected output (model-dependent):
#     ========================================================================
#     HUGGING FACE LOCAL MODEL EXAMPLE
#     ========================================================================
#     Model:     Qwen/Qwen3-0.6B
#     Device:    cpu
#     Max steps: 10
#     Agent:     Explorer at (1, 0), Goal at (4, 0)
#     Chain-of-thought: enabled
#     Recording to: experiments/logs/huggingface_goal_seeking_20260707_120000.pkl
#
#     Step-by-step trace:
#     ------------------------------------------------------------------------
#       Step 0:  x=1 -> x=2
#         Action: Move Right.
#         Reward: -0.05
#         Reasoning: [model's step-by-step thinking]
#         Raw:    {"action_choice_idx": 2}
#         ... (more steps) ...
#
#     Summary:
#       Final position:  (4, 0)
#       Goal position:   (4, 0)
#       Goal reached:    True
#       Cumulative reward: 0.85
#
#     Your episode has been recorded to:
#       experiments/logs/huggingface_goal_seeking_20260707_120000.pkl
#
#     Opening replay viewer...
#       SPACE: play/pause    LEFT/RIGHT: step    HOME/END: jump    ESC: exit
#
# The first run will download the model, which may take a minute. Subsequent
# runs use the Hugging Face disk cache.
#
# After the LLM episode completes (or the 10-step TimeLimit fires), a pygame
# window opens showing the recorded run. You can step through frame by frame
# or press SPACE to watch it in real time.
#
# WHAT TO TRY NEXT
# ================
# 1. Change model_name to a different Hugging Face model.
# 2. Move the Goal further away (change Position_2D(4, 0) to Position_2D(10, 0)).
# 3. Set use_chain_of_thought=False to see how the model performs without it.
# 4. Add obstacles or walls (use Collidable and tags=["wall"]).
# 5. Give the agent an Inventory and Pick_Up_Item actions.
# 6. Add Start_Public_Conversation / Start_Private_Conversation for
#    multi-agent communication.
# 7. Load a previously recorded pickle with:
#        python -c "from word_play.presets.renderers import replay; replay('experiments/logs/huggingface_goal_seeking_newest.pkl')"
