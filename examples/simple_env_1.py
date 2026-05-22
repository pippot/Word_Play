import argparse
import pprint
from typing import Any, Mapping, Sequence

from word_play.core import (
    Action,
    Entity,
    Target_Is_Self,
)
from word_play.presets.action_args import (
    Dict_Arg,
    Int_Arg,
    List_Arg,
    String_Arg,
    String_Choice_Arg,
)
from word_play.presets.action_policies.batching import build_policy_step_actions
from word_play.presets.action_policies.follow_action_sequence import Follow_Action_Sequence
from word_play.presets.action_policies.human import Human_Takes_Action
from word_play.presets.action_policies.llm_action_and_communication import LLM_Action_And_Communication_Policy
from word_play.presets.entity_orderings import randomize_agent_order
from word_play.presets.environments.simple_2d_grid_world import Simple_2D_Grid_World
from word_play.presets.models import Chat_Message, Model
from word_play.presets.models.registry import LLM_MODEL_REGISTRY
from word_play.presets.movement.simple_2d_grid import (
    Collidable,
    Move_Up,
    Move_Down,
    Move_Left,
    Move_Right,
    Position_2D,
)
from word_play.presets.systems.combat import Attack
from word_play.presets.systems.communication import (
    Human_Communication_Policy,
    Start_Private_Conversation,
    Start_Public_Conversation,
    TalkingCow,
)
from word_play.presets.systems.do_nothing import Do_Nothing
from word_play.presets.systems.health import Health
from word_play.presets.systems.inventory import Inventory
from word_play.utils import tilemap_to_entities


class Test_Action(Action):

    def __init__(self):
        super().__init__(
            validation_rules=[Target_Is_Self()],
            required_kwargs={
                "gold coin count": Int_Arg(),
                "assignment": Dict_Arg(String_Arg(), Int_Arg()),
                "damages": List_Arg(Int_Arg()),
                "item": String_Choice_Arg({"hammer", "apple"}),
            },
        )

    def exec_action(self, actor, target_entity, env, kwargs):
        print("=======")
        pprint.pprint(kwargs, sort_dicts=False)
        print("=======")

    def action_description_text(self, actor, target_entity, env):
        return "Test Action."


class Batch_Debug_Model(Model):
    def generate_chat(
        self,
        messages: Sequence[Chat_Message | Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        raise AssertionError("Batch debug expects the batched model path.")

    def generate_text_batch(
        self,
        input_texts: Sequence[str],
        generation_config: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        print(f"BATCH SIZE: {len(input_texts)}")
        return ['{"action_choice_idx": 1}' for _ in input_texts]


def _agent_policy(batch_debug: bool):
    if batch_debug:
        return LLM_Action_And_Communication_Policy(model_key="batch-debug")
    return Human_Takes_Action()


def _register_batch_debug_model() -> None:
    LLM_MODEL_REGISTRY.unload("batch-debug")
    LLM_MODEL_REGISTRY.register("batch-debug", Batch_Debug_Model)


def run_exp(exp_steps: int = 1000, batch_debug: bool = False):
    if batch_debug:
        _register_batch_debug_model()

    entity_tilemap = """
    WWWWWWWW...
    Wb.h...W...
    WAfs.t.WWWW
    WIc..b....W
    WWWWWWWWWWW
    """
    entity_tileset = {
        "W": {
            "name": "Wall",
            "tags": ["wall"],
            "components": [Collidable()],
        },
        "I": {
            "name": "Iskandar",
            "actions": [
                Test_Action(),
                Do_Nothing(),
                Move_Up(),
                Move_Down(),
                Move_Left(),
                Move_Right(),
                Attack(name="Zap", damage_amount=1),
                Start_Public_Conversation(),
                Start_Private_Conversation(),
            ],
            "components": [
                _agent_policy(batch_debug),
                Inventory(
                    collectable_tags=["item"],
                    inventory_size=2,
                    starting_inventory=[Entity(name="Strawberry", position=Position_2D(100, 100), tags=["item"])],
                ),
                Health(max_health=5, starting_health=3),
                Collidable(collidable_tags=["wall"]),
                Human_Communication_Policy(),
            ],
        },
        "A": {
            "name": "Andrei",
            "actions": [
                Test_Action(),
                Do_Nothing(),
                Move_Up(),
                Move_Down(),
                Move_Left(),
                Move_Right(),
                Attack(name="Zap", damage_amount=1),
                Start_Public_Conversation(),
                Start_Private_Conversation(),
            ],
            "components": [
                _agent_policy(batch_debug),
                Inventory(
                    collectable_tags=["item"],
                    inventory_size=2,
                    starting_inventory=[Entity(name="Strawberry", position=Position_2D(100, 100), tags=["item"])],
                ),
                Health(max_health=5, starting_health=3),
                Collidable(collidable_tags=["wall"]),
                Human_Communication_Policy(),
            ],
        },
        "f": {
            "name": "Blue Flower",
            "tags": ["item"],
        },
        "b": {
            "name": "Barrel",
            "tags": ["item"],
            "components": [Health(max_health=1, starting_health=1)],
        },
        "c": {
            "name": "Cow",
            "actions": [Move_Up(), Move_Down()],
            "components": [
                Health(max_health=5, starting_health=5),
                Follow_Action_Sequence([(Move_Up, None), (Move_Down, None)]),
                TalkingCow(),
            ],
        },
        "h": {
            "name": "Fat Cow",
            "components": [
                Health(max_health=10, starting_health=10),
                TalkingCow(),
            ],
        },
        "t": {
            "name": "Tiny Cow",
            "components": [
                Health(max_health=1, starting_health=1),
                TalkingCow(),
            ],
        },
        "s": {
            "name": "Spike",
            "actions": [Attack(name="Poke", damage_amount=1)],
            "tags": ["item"],
            "components": [Follow_Action_Sequence([(Attack, None)])],
        },
    }

    env = Simple_2D_Grid_World(
        description="The forbidden forest.",
        entities=tilemap_to_entities(entity_tilemap, entity_tileset),
        entity_order=randomize_agent_order,
        observation_radius=1,
    )

    for step in range(exp_steps):
        cur_step_actions = build_policy_step_actions(
            env,
            batched=True,
            on_selection=lambda env, observation, agent_id, action, info: print(
                f"[step {step}] {env.agents[agent_id].name} -> {action}"
            ),
        )

        env.step(cur_step_actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the two-agent simple env.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-debug", action="store_true")
    args = parser.parse_args()

    default_steps = 3 if args.batch_debug else 1000
    run_exp(exp_steps=args.steps or default_steps, batch_debug=args.batch_debug)
