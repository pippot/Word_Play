from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages
from word_play.presets.models.registry import LLM_MODEL_REGISTRY, Model_Registry


class Gemini_Model(Model):
    """
    Chat model backed by the Google Gen AI SDK.

    API keys are never passed through code. Set the environment variable named
    by api_key_env, which defaults to GEMINI_API_KEY.
    """

    _CLIENT_CACHE: dict[str, Any] = {}

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        generation_config: Mapping[str, Any] | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        verbosity: int = 0,
    ):
        super().__init__(verbosity=verbosity)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.default_generation_config = dict(generation_config or {})
        self.api_key_env = api_key_env

    def _get_client(self):
        if self.api_key_env in self._CLIENT_CACHE:
            return self._CLIENT_CACHE[self.api_key_env]

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Gemini_Model requires the 'google-genai' package.") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing environment variable: {self.api_key_env}")

        client = genai.Client(api_key=api_key)
        self._CLIENT_CACHE[self.api_key_env] = client
        return client

    def _build_contents(
        self, messages: Sequence[Chat_Message | Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        normalized = normalize_chat_messages(messages)
        system_parts = [self.system_prompt] if self.system_prompt else []
        contents: list[dict[str, Any]] = []

        for message in normalized:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_parts.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            part = {"text": content}
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": gemini_role, "parts": [part]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        system = "\n\n".join(part for part in system_parts if part).strip() or None
        return contents, system

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        params = {**self.default_generation_config, **(generation_config or {})}
        if max_new_tokens is not None:
            params.setdefault("max_output_tokens", max_new_tokens)

        contents, system = self._build_contents(messages)
        if system and "system_instruction" not in params:
            params["system_instruction"] = system

        response = self._get_client().models.generate_content(
            model=self.model_name,
            contents=contents,
            config=params or None,
        )
        return (getattr(response, "text", "") or "").strip()


def register_gemini_model(
    model_key: str,
    *,
    model_name: str,
    generation_config: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    api_key_env: str = "GEMINI_API_KEY",
    verbosity: int = 0,
    registry: Model_Registry | None = None,
    replace: bool = False,
) -> None:
    """
    Register a Gemini model by environment variable name.

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
        Gemini_Model,
        model_name=model_name,
        system_prompt=system_prompt,
        generation_config=generation_config,
        api_key_env=api_key_env,
        verbosity=verbosity,
    )
