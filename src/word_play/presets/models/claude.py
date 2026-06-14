from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages
from word_play.presets.models.registry import LLM_MODEL_REGISTRY, Model_Registry


class Claude_Model(Model):
    """
    Chat model backed by Anthropic's Claude API.

    API keys are never passed through code. Set the environment variable named
    by api_key_env, which defaults to ANTHROPIC_API_KEY.
    """

    _CLIENT_CACHE: dict[tuple[str, str | None], Any] = {}

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        generation_config: Mapping[str, Any] | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        default_max_tokens: int = 1024,
        verbosity: int = 0,
    ):
        super().__init__(verbosity=verbosity)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.default_generation_config = dict(generation_config or {})
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.default_max_tokens = default_max_tokens

    def _get_client(self):
        cache_key = (self.api_key_env, self.base_url)
        if cache_key in self._CLIENT_CACHE:
            return self._CLIENT_CACHE[cache_key]

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Claude_Model requires the 'anthropic' package.") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing environment variable: {self.api_key_env}")

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = Anthropic(**client_kwargs)
        self._CLIENT_CACHE[cache_key] = client
        return client

    def _build_request_messages(
        self, messages: Sequence[Chat_Message | Mapping[str, Any]]
    ) -> tuple[list[dict[str, str]], str | None]:
        normalized = normalize_chat_messages(messages)
        system_parts = [self.system_prompt] if self.system_prompt else []
        request_messages: list[dict[str, str]] = []

        for message in normalized:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_parts.append(content)
                continue

            claude_role = "assistant" if role == "assistant" else "user"
            if request_messages and request_messages[-1]["role"] == claude_role:
                request_messages[-1]["content"] += f"\n\n{content}"
            else:
                request_messages.append({"role": claude_role, "content": content})

        if not request_messages:
            request_messages.append({"role": "user", "content": ""})

        system = "\n\n".join(part for part in system_parts if part).strip() or None
        return request_messages, system

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        params = {**self.default_generation_config, **(generation_config or {})}
        params.setdefault("max_tokens", max_new_tokens or self.default_max_tokens)

        request_messages, system = self._build_request_messages(messages)
        if system and "system" not in params:
            params["system"] = system

        response = self._get_client().messages.create(
            model=self.model_name,
            messages=request_messages,
            **params,
        )

        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts).strip()


def register_claude_model(
    model_key: str,
    *,
    model_name: str,
    generation_config: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    api_key_env: str = "ANTHROPIC_API_KEY",
    base_url: str | None = None,
    default_max_tokens: int = 1024,
    verbosity: int = 0,
    registry: Model_Registry | None = None,
    replace: bool = False,
) -> None:
    """
    Register a Claude model by environment variable name.

    The API key value is intentionally not accepted here. Store it in the
    environment variable named by api_key_env.
    """
    target_registry = registry or LLM_MODEL_REGISTRY

    if model_key in target_registry:
        if not replace:
            return
        target_registry.unload(model_key)

    target_registry.register(
        model_key,
        Claude_Model,
        model_name=model_name,
        system_prompt=system_prompt,
        generation_config=generation_config,
        api_key_env=api_key_env,
        base_url=base_url,
        default_max_tokens=default_max_tokens,
        verbosity=verbosity,
    )
