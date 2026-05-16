from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages
from word_play.presets.models.registry import LLM_MODEL_REGISTRY, Model_Registry


class OpenRouter_Model(Model):
    """
    Chat model backed by OpenRouter (or any OpenAI-compatible endpoint via base_url).

    API keys are never passed through code. Set the environment variable named
    by api_key_env, which defaults to OPENROUTER_API_KEY.
    """

    _CLIENT_CACHE: dict[tuple[str | None, str, tuple[tuple[str, str], ...]], Any] = {}

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        generation_config: Mapping[str, Any] | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        verbosity: int = 0,
    ):
        super().__init__(verbosity=verbosity)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.default_generation_config = dict(generation_config or {})
        self.base_url = base_url
        self.api_key_env = api_key_env

        self._headers: dict[str, str] = {}
        if site_url:
            self._headers["HTTP-Referer"] = site_url
        if app_name:
            self._headers["X-Title"] = app_name

    def _get_client(self):
        cache_key = (self.base_url, self.api_key_env, tuple(sorted(self._headers.items())))
        if cache_key in self._CLIENT_CACHE:
            return self._CLIENT_CACHE[cache_key]

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenRouter_Model requires the 'openai' package.") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing environment variable: {self.api_key_env}")

        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=self._headers or None,
        )
        self._CLIENT_CACHE[cache_key] = client
        return client

    def _build_messages(
        self, messages: Sequence[Chat_Message | Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        normalized = normalize_chat_messages(messages)
        if self.system_prompt and not any(message["role"] == "system" for message in normalized):
            return [{"role": "system", "content": self.system_prompt}, *normalized]
        return normalized

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        params = {**self.default_generation_config, **(generation_config or {})}
        if max_new_tokens is not None:
            if self.model_name.startswith(("openai/gpt-", "openai/o")):
                params["max_completion_tokens"] = max_new_tokens
            else:
                params["max_tokens"] = max_new_tokens

        extra_body = dict(params.pop("extra_body", {}) or {})
        for openrouter_param in ("plugins", "provider", "reasoning", "transforms"):
            if openrouter_param in params:
                extra_body[openrouter_param] = params.pop(openrouter_param)
        if extra_body:
            params["extra_body"] = extra_body

        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(messages),
            **params,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


def register_openrouter_model(
    model_key: str,
    *,
    model_name: str,
    generation_config: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    site_url: str | None = None,
    app_name: str | None = None,
    verbosity: int = 0,
    registry: Model_Registry | None = None,
    replace: bool = False,
) -> None:
    """
    Register an OpenRouter model by environment variable name.

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
        OpenRouter_Model,
        model_name=model_name,
        system_prompt=system_prompt,
        generation_config=generation_config,
        site_url=site_url,
        app_name=app_name,
        base_url=base_url,
        api_key_env=api_key_env,
        verbosity=verbosity,
    )
