from __future__ import annotations

from typing import Any, Mapping, Sequence

from word_play.presets.models.model import Chat_Message, Model, normalize_chat_messages


class Human_Model(Model):
    """
    Replaces the LLM with a human typing responses.
    The human sees the exact same prompt the LLM would receive.
    """

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        normalized = normalize_chat_messages(messages)
        rendered = "\n".join(f"[{message['role']}] {message['content']}" for message in normalized)
        return input(
            f"\n========================= input START =========================\n"
            f"{rendered}\n"
            f"HUMAN input: "
        )

    def generate_chat_batch(
        self,
        messages_batch: Sequence[Sequence[Chat_Message | Mapping[str, Any]]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        return [
            self.generate_chat(messages, generation_config=generation_config, max_new_tokens=max_new_tokens)
            for messages in messages_batch
        ]
