from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages
from word_play.presets.models.registry import LLM_MODEL_REGISTRY, Model_Registry


class HuggingFace_Model(Model):
    """
    Local Hugging Face Transformers text-generation model.

    Public models do not need a key. For private Hub models, set the environment
    variable named by token_env, which defaults to HF_TOKEN.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        generation_config: Mapping[str, Any] | None = None,
        tokenizer_name: str | None = None,
        task: str = "text-generation",
        token_env: str | None = "HF_TOKEN",
        pipeline_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        device: int | str | None = None,
        verbosity: int = 0,
    ):
        super().__init__(verbosity=verbosity)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.default_generation_config = dict(generation_config or {})
        self.tokenizer_name = tokenizer_name
        self.task = task
        self.token_env = token_env
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.device = device
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("HuggingFace_Model requires the 'transformers' package.") from exc

        kwargs = dict(self.pipeline_kwargs)
        if self.tokenizer_name:
            kwargs["tokenizer"] = self.tokenizer_name
        if self.model_kwargs:
            kwargs["model_kwargs"] = self.model_kwargs
        if self.device is not None:
            kwargs["device"] = self.device
        if self.token_env:
            token = os.getenv(self.token_env)
            if token:
                kwargs["token"] = token

        self._pipeline = pipeline(self.task, model=self.model_name, **kwargs)
        return self._pipeline

    def _build_messages(
        self, messages: Sequence[Chat_Message | Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        normalized = normalize_chat_messages(messages)
        if self.system_prompt and not any(message["role"] == "system" for message in normalized):
            return [{"role": "system", "content": self.system_prompt}, *normalized]
        return normalized

    def _extract_generated_text(self, output: Any) -> str:
        if isinstance(output, list) and output:
            output = output[0]

        if isinstance(output, dict):
            generated_text = output.get("generated_text", output.get("text", ""))
        else:
            generated_text = output

        if isinstance(generated_text, list) and generated_text:
            for message in reversed(generated_text):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    return str(message.get("content", "")).strip()
            last_message = generated_text[-1]
            if isinstance(last_message, dict):
                return str(last_message.get("content", "")).strip()

        if generated_text is None:
            return ""
        return str(generated_text).strip()

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        params = {**self.default_generation_config, **(generation_config or {})}
        if max_new_tokens is not None:
            params.setdefault("max_new_tokens", max_new_tokens)
        params.setdefault("return_full_text", False)

        output = self._get_pipeline()(self._build_messages(messages), **params)
        return self._extract_generated_text(output)


def register_huggingface_model(
    model_key: str,
    *,
    model_name: str,
    generation_config: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    tokenizer_name: str | None = None,
    task: str = "text-generation",
    token_env: str | None = "HF_TOKEN",
    pipeline_kwargs: Mapping[str, Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    device: int | str | None = None,
    verbosity: int = 0,
    registry: Model_Registry | None = None,
    replace: bool = False,
) -> None:
    """
    Register a Hugging Face model.

    Token values are intentionally not accepted here. For private Hub models,
    store the token in the environment variable named by token_env.
    """
    target_registry = registry or LLM_MODEL_REGISTRY

    if model_key in target_registry:
        if not replace:
            return
        target_registry.unload(model_key)

    target_registry.register(
        model_key,
        HuggingFace_Model,
        model_name=model_name,
        system_prompt=system_prompt,
        generation_config=generation_config,
        tokenizer_name=tokenizer_name,
        task=task,
        token_env=token_env,
        pipeline_kwargs=pipeline_kwargs,
        model_kwargs=model_kwargs,
        device=device,
        verbosity=verbosity,
    )
