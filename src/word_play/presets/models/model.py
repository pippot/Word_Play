from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


Chat_Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True, frozen=True)
class Chat_Message:
    role: Chat_Role
    content: str


def normalize_chat_messages(messages: Sequence[Chat_Message | Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, Chat_Message):
            normalized_messages.append({"role": message.role, "content": message.content})
            continue

        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if not role:
            raise ValueError(f"Chat message is missing a role: {message}")
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


class Model(ABC):
    """
    Provider-agnostic model interface.

    Preferred method:
        - generate_chat(messages=[...])
        - generate_chat_batch(messages_batch=[[...], ...])

    Backward compatibility:
        - generate_text(input_text=...) remains available and maps to generate_chat.
        - generate_text(input_text=[...]) maps to generate_text_batch.
    """

    def __init__(self, verbosity: int = 0) -> None:
        self.verbosity = verbosity

    @abstractmethod
    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        pass

    def generate_chat_batch(
        self,
        messages_batch: Sequence[Sequence[Chat_Message | Mapping[str, Any]]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        if not messages_batch:
            return []

        if len(messages_batch) == 1:
            return [self.generate_chat(messages_batch[0], generation_config, max_new_tokens)]

        worker_count = max_workers or min(32, len(messages_batch))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(
                executor.map(
                    lambda messages: self.generate_chat(messages, generation_config, max_new_tokens),
                    messages_batch,
                )
            )

    def generate_text_batch(
        self,
        input_texts: Sequence[str],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        return self.generate_chat_batch(
            messages_batch=[[Chat_Message(role="user", content=input_text)] for input_text in input_texts],
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            max_workers=max_workers,
        )

    def generate_text(
        self,
        input_text: str | list[str],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
        max_workers: int | None = None,
    ) -> str | list[str]:
        if isinstance(input_text, str):
            return self.generate_chat(
                messages=[Chat_Message(role="user", content=input_text)],
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            )

        return self.generate_text_batch(
            input_texts=input_text,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            max_workers=max_workers,
        )

    def cond_logP(self, inputs: str | list[str], targets: list[str] | list[list[str]]) -> float | list[float]:
        raise NotImplementedError()
