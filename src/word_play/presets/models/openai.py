from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages
from word_play.presets.models.registry import LLM_MODEL_REGISTRY, Model_Registry


class OpenAI_Model(Model):
    """
    Chat model backed by OpenAI.

    API keys are never passed through code. Set the environment variable named
    by api_key_env, which defaults to OPENAI_API_KEY.
    """

    _CLIENT_CACHE: dict[tuple[str | None, str, str | None, str | None], Any] = {}

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        generation_config: Mapping[str, Any] | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        organization: str | None = None,
        project: str | None = None,
        verbosity: int = 0,
    ):
        super().__init__(verbosity=verbosity)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.default_generation_config = dict(generation_config or {})
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.organization = organization
        self.project = project

    def _get_client(self):
        cache_key = (self.base_url, self.api_key_env, self.organization, self.project)
        if cache_key in self._CLIENT_CACHE:
            return self._CLIENT_CACHE[cache_key]

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI_Model requires the 'openai' package.") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing environment variable: {self.api_key_env}")

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.organization:
            client_kwargs["organization"] = self.organization
        if self.project:
            client_kwargs["project"] = self.project

        client = OpenAI(**client_kwargs)
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
            params.setdefault("max_completion_tokens", max_new_tokens)

        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(messages),
            **params,
        )
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


def register_openai_model(
    model_key: str,
    *,
    model_name: str,
    generation_config: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    organization: str | None = None,
    project: str | None = None,
    verbosity: int = 0,
    registry: Model_Registry | None = None,
    replace: bool = False,
) -> None:
    """
    Register an OpenAI model by environment variable name.

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
        OpenAI_Model,
        model_name=model_name,
        system_prompt=system_prompt,
        generation_config=generation_config,
        base_url=base_url,
        api_key_env=api_key_env,
        organization=organization,
        project=project,
        verbosity=verbosity,
    )
