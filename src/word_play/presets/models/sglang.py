"""
SGLang model preset for Word Play.

This module provides :class:`SGLang_Model`, a chat-completions client that
talks to a running SGLang inference server.

WHAT IS SGLANG
==============
SGLang (https://github.com/sgl-project/sglang) is a high-performance serving
framework for large language models. It exposes an OpenAI-compatible HTTP
API on top of its RadixAttention engine, so any client that speaks the
OpenAI chat-completions protocol can talk to it.

Because SGLang's wire format is identical to OpenAI's, this preset reuses
the ``openai`` Python package to issue requests. No new client library is
required.

STARTING AN SGLANG SERVER
=========================
Before running any Word Play example that uses ``SGLang_Model``, launch
SGLang's HTTP server in another terminal. The most common form is::

    python -m sglang.launch_server \\
        --model-path meta-llama/Llama-3.1-8B-Instruct \\
        --port 30000 \\
        --host 0.0.0.0

Once the server prints a "Launch success" line and reports
``http://localhost:30000`` as the OpenAI-compatible endpoint, the preset
will be able to reach it. The default ``base_url`` used by
:class:`SGLang_Model` matches this default.

AUTHENTICATION
==============
SGLang does not require an API key by default. If you started the server
with ``--api-key some-secret``, the client must send the same value in its
``Authorization`` header. Set it via the ``api_key_env`` parameter
(defaults to the ``SGLANG_API_KEY`` environment variable). For local
servers that do not enforce authentication, the preset falls back to the
placeholder string ``"EMPTY"``, which is SGLang's conventional dummy
token for unauthenticated local servers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports.
# ``os`` is needed to read the API-key environment variable.
# ``Any``, ``Mapping`` and ``Sequence`` are typing helpers used throughout
# the presets module.
# ---------------------------------------------------------------------------
import os
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Word Play core model types.
# - ``Model``            : the abstract base class every model preset inherits.
# - ``Chat_Message``     : the small dataclass used to type individual
#                          messages in a chat (role + content).
# - ``normalize_chat_messages`` : utility that coerces ``Chat_Message`` and
#                          raw ``{"role", "content"}`` dicts into a uniform
#                          list of OpenAI-style dicts.
# ---------------------------------------------------------------------------
from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages

# ---------------------------------------------------------------------------
# The registry that turns string keys (held by policies) into live model
# instances. We need it so the ``register_sglang_model`` helper can store
# the spec lazily, the same way the other presets do.
# ---------------------------------------------------------------------------
from word_play.presets.models.registry import LLM_MODEL_REGISTRY, Model_Registry


class SGLang_Model(Model):
    """
    Chat model backed by an SGLang inference server.

    This preset uses the ``openai`` Python client to hit SGLang's
    OpenAI-compatible ``/v1/chat/completions`` endpoint, so it works with
    any SGLang server without extra dependencies.

    The default ``base_url`` (``http://localhost:30000/v1``) matches the
    SGLang server's default port. Override it if you serve on a different
    host/port.

    Parameters
    ----------
    model_name : str
        The model identifier the SGLang server was started with
        (for example ``"meta-llama/Llama-3.1-8B-Instruct"`` or
        ``"Qwen/Qwen2.5-7B-Instruct"``). Pass the same value you gave to
        ``--model-path`` when launching the server.
    system_prompt : str, optional
        Default system prompt prepended to every chat. If the caller
        already includes a system message, that one wins.
    generation_config : Mapping[str, Any], optional
        Default sampling/chat parameters forwarded to SGLang
        (e.g. ``{"temperature": 0.0, "top_p": 1.0}``). Per-call
        ``generation_config`` arguments override these.
    base_url : str, optional
        OpenAI-compatible base URL of the SGLang server. Defaults to
        ``http://localhost:30000/v1``.
    api_key_env : str | None, optional
        Name of the environment variable holding the API key. Set to
        ``None`` to never read the environment (the preset will then
        always send the placeholder ``"EMPTY"``).
    default_headers : Mapping[str, str], optional
        Extra HTTP headers to attach to every request. Useful when
        running behind a reverse proxy that requires custom auth headers.
    verbosity : int, optional
        Verbosity flag forwarded to the base ``Model`` class.
    """

    # ------------------------------------------------------------------
    # Cached OpenAI clients keyed by (base_url, api_key_env, headers).
    # The OpenAI client is thread-safe and holds an underlying HTTP
    # connection pool, so reusing one client across many calls is the
    # right default. Caching by connection parameters means that a
    # second ``SGLang_Model`` pointing at the same server can share the
    # same client.
    # ------------------------------------------------------------------
    _CLIENT_CACHE: dict[tuple[str, str, tuple[tuple[str, str], ...]], Any] = {}

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        generation_config: Mapping[str, Any] | None = None,
        base_url: str = "http://localhost:30000/v1",
        api_key_env: str | None = "SGLANG_API_KEY",
        default_headers: Mapping[str, str] | None = None,
        verbosity: int = 0,
    ):
        # Initialize the base Model so ``self.verbosity`` is set, exactly
        # like every other preset does.
        super().__init__(verbosity=verbosity)

        # Store the model identifier the SGLang server is serving.
        self.model_name = model_name

        # Store the default system prompt (may be empty).
        self.system_prompt = system_prompt

        # Copy the default sampling config so later per-call mutations
        # cannot leak across instances.
        self.default_generation_config = dict(generation_config or {})

        # OpenAI-compatible base URL of the running SGLang server.
        self.base_url = base_url

        # Name of the environment variable that holds the API key, or
        # ``None`` to skip the env lookup entirely.
        self.api_key_env = api_key_env

        # Optional custom headers (e.g. for reverse proxies).
        self._headers: dict[str, str] = dict(default_headers or {})

    def _get_client(self):
        """
        Lazily build (and cache) the OpenAI client pointed at the
        SGLang server.
        """
        # Build a deterministic cache key from the connection parameters.
        cache_key = (
            self.base_url,
            self.api_key_env or "",
            tuple(sorted(self._headers.items())),
        )
        if cache_key in self._CLIENT_CACHE:
            return self._CLIENT_CACHE[cache_key]

        # The OpenAI Python client is an optional dependency. We import
        # it lazily so projects that do not use ``SGLang_Model`` do not
        # pay the import cost.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "SGLang_Model requires the 'openai' package. "
                "Install it via `pip install -r optional_requirements.txt`."
            ) from exc

        # Resolve the API key. The OpenAI client requires *some* string
        # for ``api_key``, but SGLang does not actually validate the
        # value unless the server was launched with ``--api-key``. The
        # conventional placeholder is the literal string ``"EMPTY"``.
        api_key = "EMPTY"
        if self.api_key_env:
            env_value = os.getenv(self.api_key_env)
            if env_value:
                api_key = env_value

        # Build the client. ``base_url`` is what tells the OpenAI client
        # to talk to SGLang instead of api.openai.com.
        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=self._headers or None,
        )

        # Cache and return.
        self._CLIENT_CACHE[cache_key] = client
        return client

    def _build_messages(
        self, messages: Sequence[Chat_Message | Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        """
        Coerce a sequence of mixed messages into OpenAI/SGLang format and
        inject the default system prompt if needed.
        """
        # Normalize to the standard ``{"role": ..., "content": ...}`` shape.
        normalized = normalize_chat_messages(messages)

        # If we have a default system prompt and the caller has not
        # already supplied one, prepend it.
        if self.system_prompt and not any(
            message["role"] == "system" for message in normalized
        ):
            return [{"role": "system", "content": self.system_prompt}, *normalized]
        return normalized

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """
        Call the SGLang server and return the assistant's text.

        Parameters
        ----------
        messages : Sequence[Chat_Message | Mapping[str, Any]]
            The chat history. Anything in ``Chat_Message`` form or raw
            OpenAI-style ``{"role", "content"}`` dicts is accepted.
        generation_config : Mapping[str, Any], optional
            Per-call overrides for sampling parameters (temperature,
            top_p, response_format, ...). Anything accepted by
            OpenAI's chat-completions endpoint is accepted here.
        max_new_tokens : int, optional
            Maximum number of tokens to generate. Mapped to OpenAI's
            standard ``max_tokens`` parameter, which is exactly what
            SGLang expects.

        Returns
        -------
        str
            The assistant's textual reply, with surrounding whitespace
            stripped.
        """
        # Merge the per-call config on top of the instance defaults so
        # later overrides win.
        params: dict[str, Any] = {
            **self.default_generation_config,
            **(generation_config or {}),
        }

        # SGLang follows the original OpenAI chat-completions schema, so
        # ``max_tokens`` is the right field. (``max_completion_tokens``
        # is the newer OpenAI-specific field used by some recent
        # OpenAI-only models; SGLang does not require it.)
        if max_new_tokens is not None:
            params.setdefault("max_tokens", max_new_tokens)

        # SGLang-specific sampling parameters (e.g. ``top_k``, ``min_p``,
        # ``repetition_penalty``) are not part of the OpenAI spec, but
        # SGLang accepts them through the ``extra_body`` field of the
        # OpenAI client. We forward whatever the caller put there.
        extra_body = dict(params.pop("extra_body", {}) or {})

        # Disable Qwen3-style "thinking" output by default. Qwen3's chat
        # template emits a ``<think>...</think>`` block followed by the
        # actual reply when ``enable_thinking`` is left on, and SGLang
        # folds that block into ``message.content`` -- so without this,
        # the LLM's internal reasoning leaks into the visible response.
        # Callers can override by passing their own ``chat_template_kwargs``
        # in ``extra_body``.
        if "chat_template_kwargs" not in extra_body:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        params["extra_body"] = extra_body

        # Hit the server.
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(messages),
            **params,
        )

        # Extract the assistant text from the response. SGLang mirrors
        # the OpenAI schema, so ``response.choices[0].message.content``
        # is the right path. Some servers return structured parts
        # (a list of dicts/objects) instead of a plain string, so we
        # handle both shapes defensively.
        content = response.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            ).strip()
        return str(content).strip()


def register_sglang_model(
    model_key: str,
    *,
    model_name: str,
    generation_config: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    base_url: str = "http://localhost:30000/v1",
    api_key_env: str | None = "SGLANG_API_KEY",
    default_headers: Mapping[str, str] | None = None,
    verbosity: int = 0,
    registry: Model_Registry | None = None,
    replace: bool = False,
) -> None:
    """
    Register an SGLang model in :data:`LLM_MODEL_REGISTRY` (or a custom
    registry).

    Like every other ``register_*_model`` helper in this package, the
    function stores a spec (the class plus its constructor kwargs) in the
    registry. The actual :class:`SGLang_Model` instance is not built
    until the first agent calls ``LLM_MODEL_REGISTRY.resolve(model_key)``.

    The API key value is intentionally not accepted here. If the SGLang
    server was started with ``--api-key``, store the value in the
    environment variable named by ``api_key_env``. For local servers
    without authentication, leave ``api_key_env`` at its default and the
    preset will use the placeholder string ``"EMPTY"``.

    Parameters
    ----------
    model_key : str
        The string key agents will pass in
        :class:`word_play.presets.action_policies.llm_action_and_communication.LLM_Action_And_Communication_Policy`.
    model_name : str
        The model identifier the SGLang server was started with.
    generation_config : Mapping[str, Any], optional
        Default sampling parameters forwarded to SGLang on every call.
    system_prompt : str, optional
        Default system prompt prepended to every chat.
    base_url : str, optional
        OpenAI-compatible base URL of the SGLang server.
    api_key_env : str | None, optional
        Name of the environment variable holding the API key, or
        ``None`` to skip the lookup.
    default_headers : Mapping[str, str], optional
        Extra HTTP headers to attach to every request.
    verbosity : int, optional
        Verbosity flag forwarded to the model.
    registry : Model_Registry, optional
        Custom registry to use. Defaults to the global
        :data:`LLM_MODEL_REGISTRY`.
    replace : bool, optional
        If ``True`` and a spec under ``model_key`` already exists, unload
        the existing instance and replace it. The default ``False`` makes
        registration a no-op when the key is already taken.
    """
    # Use the caller's registry if provided, otherwise the global one.
    target_registry = registry or LLM_MODEL_REGISTRY

    # If the key is taken and the caller did not opt in to replacement,
    # silently keep the existing spec. This mirrors the behaviour of the
    # other ``register_*_model`` helpers in this package.
    if model_key in target_registry:
        if not replace:
            return
        target_registry.unload(model_key)

    # Store the spec. The class is constructed lazily on the next
    # ``resolve(model_key)`` call from any agent.
    target_registry.register(
        model_key,
        SGLang_Model,
        model_name=model_name,
        system_prompt=system_prompt,
        generation_config=generation_config,
        base_url=base_url,
        api_key_env=api_key_env,
        default_headers=default_headers,
        verbosity=verbosity,
    )
