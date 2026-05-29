from __future__ import annotations

import threading
import time
import unittest
from typing import Any, Mapping, Sequence

from word_play.core import Entity
from word_play.presets.action_policies.batching import build_policy_step_actions
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import Chat_Message, Model
from word_play.presets.models.registry import LLM_MODEL_REGISTRY
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.systems.do_nothing import Do_Nothing


class Thread_Debug_Model(Model):
    thread_names: list[str] = []
    lock = threading.Lock()

    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        with self.lock:
            self.thread_names.append(threading.current_thread().name)
        time.sleep(0.01)
        return '{"action_choice_idx": 0}'


class ActionBatchingTests(unittest.TestCase):
    def test_llm_agents_are_selected_with_threadpool(self) -> None:
        model_key = "test-threaded-action-model"
        LLM_MODEL_REGISTRY.unload(model_key)
        LLM_MODEL_REGISTRY.register(model_key, Thread_Debug_Model)
        Thread_Debug_Model.thread_names = []

        env = Simple_2D_Grid_World(
            description="thread batch test",
            entities=[
                Entity(
                    name="Agent A",
                    position=Position_2D(0, 0),
                    actions=[Do_Nothing()],
                    components=[LLM_Action_And_Communication_Policy(model_key=model_key)],
                ),
                Entity(
                    name="Agent B",
                    position=Position_2D(1, 0),
                    actions=[Do_Nothing()],
                    components=[LLM_Action_And_Communication_Policy(model_key=model_key)],
                ),
            ],
        )

        selection_infos = []
        selections = build_policy_step_actions(
            env,
            batched=True,
            on_selection=lambda env, observation, agent_id, selection, info: selection_infos.append(info),
        )

        self.assertEqual([str(selection) for selection in selections], ["Do nothing.", "Do nothing."])
        self.assertEqual(len(Thread_Debug_Model.thread_names), 2)
        self.assertTrue(all(name.startswith("ThreadPoolExecutor") for name in Thread_Debug_Model.thread_names))
        self.assertEqual(len(selection_infos), 2)


if __name__ == "__main__":
    unittest.main()
