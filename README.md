# Word Play

# Beta Disclaimer

Word Play is currently in public beta. The framework is stable, however, we are still collecting feedback from the community and do not yet commit to long-term backwards compatability until the full release. Please let me us know if you run into any issues, we are happy to help.

# Quick Start
Word Play is a small framework for building text-first multi-agent environments.
It is designed around plain Python environment classes, composable entity
components, and readable observations that can be consumed by humans or LLM
policies.

## Quick Start

From the repo root:

```bash
pip install -e .
python examples/simple_env_0.py
```

Some examples use optional packages. Install them with:

```bash
pip install -r optional_requirements.txt
```

The LLM examples also require an OpenRouter API key:

```bash
export OPENROUTER_API_KEY=...
python examples/agent_goal_tracking.py
python examples/llm_communication.py
```

### Install with `uv` (recommended for Linux / fresh checkouts)

`uv` is the fastest way to get a working environment on a new machine
(including Ubuntu). The repo ships a `uv.lock` so a single command resolves
and installs everything in a virtual environment:

```bash
# Install uv once (e.g. on Ubuntu):
curl -LsSf https://astral.sh/uv/install.sh | sh

# In the repo root, install word-play + LLM + renderer dependencies:
uv sync --all-extras

# Run any example inside the managed environment:
uv run python examples/simple_env_0.py
uv run python examples/llm_among_us.py
```

The optional dependency groups in `pyproject.toml` are:

| Extra    | Installs            | Used by                                |
| -------- | ------------------- | -------------------------------------- |
| `render` | `pygame`            | `rendering_demo.py`, replay tool       |
| `llm`    | `openai`            | all LLM examples                       |
| `all`    | `pygame` + `openai` | `llm_among_us.py` and the replay tool  |

To target a specific Linux architecture other than the lockfile's default
x86_64, pass `--python-platform`:

```bash
uv sync --all-extras --python-platform aarch64-unknown-linux-gnu   # ARM64
uv sync --all-extras --python-platform x86_64-unknown-linux-gnu   # x86_64
```

## Examples

- `examples/simple_env_0.py`: a minimal `Simple_2D_Grid_World` showing human-controlled actions, communication, combat, health, inventory, and collision.
- `examples/complex_tilemap.py`: a larger tilemap-driven grid world that demonstrates `tilemap_to_entities`, `randomize_agent_order`, observation radius, and typed action kwargs.
- `examples/rendering_demo.py`: a live pygame rendering demo with sprite-backed entities, walls, items, mixed human and random policies, and reset/quit handling through `env.render()`.
- `examples/parallelized_agents.py`: a renderer-backed arena that computes multiple agent actions in parallel. This is an important speed-up for multi-agent environments
- `examples/agent_goal_tracking.py`: a small one-dimensional action-only LLM example where an agent learns to move toward a goal under a time limit.
- `examples/llm_communication.py`: a three-agent LLM coordination game where each agent shares a private signal, communicates, and then chooses a synchronized action.
- `examples/llm_among_us.py`: a social-deduction 2D grid game where 6 LLM agents (1 impostor + 5 crewmates, served by SGLang / Qwen3-27B) move, chat, attack, and call emergency meetings; the run is recorded to a replay log (no live window) so it can be inspected later with the pygame replay tool.
- `examples/sglang_inference.py`: a goal-seeking agent whose LLM is served by a local [SGLang](https://github.com/sgl-project/sglang) inference server (talks to the server's OpenAI-compatible HTTP API).
- `examples/huggingface_local_model.py`: a fully local LLM agent using a Hugging Face Transformers model.

## Optional Dependencies

The base package intentionally has no runtime dependencies in `requirements.txt`.
Optional features need extra packages:

- Renderer presets: `pygame`
- OpenAI/OpenRouter/SGLang-backed LLM models: `openai` (SGLang preset also needs a running SGLang server)

Install both with:

```bash
pip install -r optional_requirements.txt
```

## Design Philosophy

Word Play environments are meant to be easy to inspect and extend. The main
building blocks are:

- `Entity`: a named object in the environment with a position, tags, actions, and components.
- `Action`: executable behavior with validation rules and optional typed kwargs.
- `Component`: reusable state or behavior attached to an entity. Components can add tags, actions, and lifecycle hooks.
- `Agent_Policy`: a component that chooses an action from an observation.
- `Communication_Policy`: a component for message passing between agents.
- `Environment`: the simulation driver. It owns entities, movement, rewards, observations, and step execution.

The environment step follows an Agent Environment Cycle-style execution model:
agent actions are selected for the current step, then executed in the current
entity order. This matters when actions conflict. For example, if two agents try
to pick up the same item, the earlier entity in the execution order can succeed
and the later one can fail. See `src/word_play/core/environment.py` for the full
step sequence.

Entity order is configurable. Use `entity_definition_order` for deterministic
definition order, `random_order` to shuffle all entities, or
`randomize_agent_order` to shuffle only agents while preserving non-agent
placement. `randomize_agent_order` is useful when turn order could otherwise
create an unfair advantage.

## Building an Environment

A typical environment defines:

1. Entities, including each agent's actions and policy components.
2. A movement system that determines position type and nearby entities.
3. A reward function returning one reward per agent.
4. An `observe(agent_id)` method that returns an `Observation`.
5. Optional start/end-of-step logic for environment-specific state.

For many grid-world examples, start with `Simple_2D_Grid_World` and
`Simple_Observation`. Add custom sections to observations with
`extra_sections` when the agent needs task-specific context.

## Presets Overview

The `src/word_play/presets` folder contains reusable pieces for common
environments:

- `action_policies`: human input, fixed action sequences, and LLM action plus communication policy.
- `models`: model interface, registry, human model, OpenAI/OpenRouter/Claude/Gemini/Hugging Face clients, and an SGLang inference preset.
- `movement`: simple 1D grid, 2D grid, single-point movement, and collision helpers.
- `observation`: `Simple_Observation` plus formatting helpers for action lists and entity state.
- `systems`: reusable gameplay systems such as communication, combat, health, inventory, action composition, and do-nothing actions.
- `environments`: ready-made environment bases such as `Simple_2D_Grid_World`.
- `renderers`: pygame-based renderer/runtime tools.
- `entity_orderings.py`: deterministic and randomized entity ordering helpers.
- `reward_functions.py`: simple reward helpers.
- `action_args.py` and `action_validations.py`: reusable typed kwargs and action validators.

For renderer architecture, extension points, and pygame implementation details,
see `src/word_play/presets/renderers/README.md`.

## Codebase Layout

- `src/word_play/core`: framework primitives: actions, components, entities, environments, movement, and observations.
- `src/word_play/presets`: reusable policies, systems, movement models, observations, models, renderers, and environment helpers.
- `src/word_play/utils`: utility helpers, including tilemap construction.
- `examples`: small runnable examples showing how to assemble environments.
- `sprite_library`: renderer sprite assets and generation utilities.
- `tests`: test package placeholder.

## Acknowledgements

We acknowledge the following sponsors, who support our research with financial and in-kind contributions: CIFAR through the Canada CIFAR AI Chair, NSERC through the Discovery Grant and an Alliance Grant with ServiceNow and DRDC, the Schmidt Sciences foundation through the AI2050 Early Career Fellow program. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute. We thank the computing team at the University of Toronto’s Computer Science Department for administrating and procuring the compute infrastructure used for the experiments in this paper.

This work was supported in part by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039B; by the Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645; by Schmidt Sciences SAFE-AI Grant; by the Frontier Model Forum and AI Safety Fund; by Coefficient Giving; by the Canadian AI Safety Institute Research Program at CIFAR; by the Canadian AI Safety Institute Research Program at CIFAR through a Catalyst Award; by the Survival and Flourishing Fund; and by the Cooperative AI Foundation. The usage of OpenAI credits is largely supported by the Tübingen AI Center and Schmidt Sciences. Resources used in preparing this research project were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

We are thankful to Darci Prout for coming up with the name "WordPlay."
