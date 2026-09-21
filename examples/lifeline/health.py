"""
Model health checks: catch a server that returns noise before -- and while --
a run burns hours of GPU time on it.

A healthy model that misformats an answer produces readable text around a
broken JSON object. A broken server (bad kernels, a quantized checkpoint
without matching kernels for the GPU, numerically corrupted state) produces
punctuation soup, which JSON mode then squeezes into JSON-like shapes. The
checks below tell the two apart, and say so in plain words.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from word_play.core import Agent_Policy
from word_play.presets.models import LLM_MODEL_REGISTRY

from .config import ACTION_GENERATION_CONFIG, REASONING_GENERATION_CONFIG
from .policy import parse_json_object
from .prompts import REASONING_INSTRUCTION

MIN_READABILITY = 0.55
TROUBLESHOOTING = "See 'Troubleshooting: garbled model output' in examples/lifeline/README.md."


class ModelHealthError(RuntimeError):
    """The model server returns unusable output; the run cannot produce valid data."""


def readability(text: str) -> float:
    """Share of non-space characters that are letters. Ordinary English prose
    scores around 0.8; the noise a broken server emits scores far lower."""
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def _snippet(text: str | None, limit: int = 160) -> str:
    text = (text or "").replace("\n", " ")
    return repr(text if len(text) <= limit else text[:limit] + "...")


def check_model(model_key: str, label: str, *, realistic: tuple[str, str] | None = None, concurrency: int = 5) -> None:
    """
    Raise ModelHealthError unless the model behind model_key
      1. answers a trivial plain-text request sensibly,
      2. returns a parseable action object in JSON mode, and
      3. writes readable reasoning for a real Lifeline prompt, `concurrency`
         times in parallel (batching bugs only show up under load).
    """
    model = LLM_MODEL_REGISTRY.resolve(model_key)
    greedy = {"temperature": 0.0}
    problems = []

    plain = model.generate_chat(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Reply with exactly one word: ready"}],
        greedy, max_new_tokens=16,
    )
    if "ready" not in (plain or "").lower():
        problems.append(f"plain request: expected 'ready', got {_snippet(plain)}")

    raw = model.generate_chat(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": 'Reply with ONLY this JSON object and nothing else: {"action_choice_idx": 2, "action_kwargs": {}}'}],
        {**ACTION_GENERATION_CONFIG, **greedy}, max_new_tokens=64,
    )
    try:
        if parse_json_object(raw).get("action_choice_idx") != 2:
            problems.append(f"JSON-mode request: wrong content {_snippet(raw)}")
    except Exception:
        problems.append(f"JSON-mode request: unparseable {_snippet(raw)}")

    if realistic is not None:
        system, user = realistic
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            replies = list(pool.map(
                lambda _: model.generate_chat(messages, REASONING_GENERATION_CONFIG, max_new_tokens=384),
                range(concurrency),
            ))
        bad = [r for r in replies if len(r or "") < 20 or readability(r) < MIN_READABILITY]
        if bad:
            problems.append(
                f"{len(bad)}/{concurrency} parallel reasoning replies to a real Lifeline prompt are not "
                f"readable text (readability {readability(bad[0]):.2f}, e.g. {_snippet(bad[0])})"
            )

    if problems:
        raise ModelHealthError(
            f"The {label} model ({model_key}) returns unusable output -- a server problem, not a "
            "Lifeline one:\n  - " + "\n  - ".join(problems) + f"\n{TROUBLESHOOTING}"
        )


def realistic_prompt(env, misaligned: bool) -> tuple[str, str] | None:
    """(system, user) of a real reasoning call for a courier or the misaligned agent in env."""
    for agent_id, agent in enumerate(env.agents):
        if env.is_misaligned(agent) == misaligned:
            policy = agent.get_component(Agent_Policy)
            return policy.system_prompt, f"{policy._context(env.observe(agent_id))}\n\n{REASONING_INSTRUCTION}"
    return None
