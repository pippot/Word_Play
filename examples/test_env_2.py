from word_play.environment import (
    Position,
    Environment,
    Entity,
    Component,
    Agent_Policy,
    Non_Agent_Policy,
    Observation,
    Action_Arg,
    String_Arg,
    Int_Arg,
    arg_in_range,
    String_Choice_Arg,
    Dynamic_Choice_Arg,
    List_Arg,
    Dict_Arg,
    Action,
    Action_Validation,
    Action_Chain,
    Target_Is_Self,
    Target_Not_Self,
    Target_Is_Nearby,
    Target_Has_Tag,
    Target_Doesnt_Have_Tag,
    Target_Has_Component,
    Action_Selection,
    entity_definition_order,
    random_order,
    randomize_agent_order,
)

from word_play.presets.movement_system_presets import (
    INFINITE_2D_MOVEMENT_SYSTEM,
    Move_Up,
    Move_Down,
    Move_Left,
    Move_Right,
    Position_2D,
    Collidable,
)
from word_play.presets.reward_func_presets import zero_reward_func
from word_play.presets.action_presets import Do_Nothing
from word_play.presets.observation_presets import format_possible_actions

from dataclasses import dataclass
from typing import Any, Callable, Iterable
from copy import deepcopy
import pprint
from abc import ABC, abstractmethod


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


def run_exp():
    exp_steps = 1000

    env = Simple_2D_Grid_World(
        description="The forbidden forest.",
        entities=[
            Entity(
                name="Iskandar",
                position=Position_2D(0, 0),
                actions=[
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
                components=[
                    Human_Takes_Action(),
                    Inventory(
                        collectable_tags=["item"],
                        inventory_size=2,
                        starting_inventory=[Entity(name="Strawberry", position=Position_2D(100, 100), tags=["item"])],
                    ),
                    Health(max_health=5, starting_health=3),
                    Collidable(collidable_tags=["wall"]),
                    Human_Communication_Policy(),
                ],
            ),
            # Entity(
            #     name="Andrei",
            #     position=Position_2D(0, 0),
            #     actions=[
            #         Do_Nothing(),
            #         Move_Up(),
            #         Move_Down(),
            #         Move_Left(),
            #         Move_Right(),
            #         Attack(name="Zap", damage_amount=1),
            #     ],
            #     components=[
            #         Human_Takes_Action(),
            #         Inventory(
            #             collectable_tags=["item"],
            #             inventory_size=2,
            #             starting_inventory=[
            #                 Entity(name="Strawberry", position=Position_2D(100, 100), tags=["item"])
            #             ],
            #         ),
            #         Health(max_health=5, starting_health=3),
            #         Collidable(collidable_tags=["wall"]),
            #     ],
            # ),
            Entity(name="Blue Flower", position=Position_2D(0, 1), tags=["item"]),
            Entity(
                name="Barrel",
                position=Position_2D(0, 0),
                tags=["item"],
                components=[Health(max_health=1, starting_health=1)],
            ),
            Entity(
                name="Barrel",
                position=Position_2D(0, 1),
                tags=["item"],
                components=[Health(max_health=1, starting_health=1)],
            ),
            Entity(
                name="Cow",
                position=Position_2D(1, 0),
                actions=[Move_Up(), Move_Down()],
                components=[
                    Health(max_health=5, starting_health=5),
                    Follow_Action_Sequence([(Move_Up, None), (Move_Down, None)]),
                    TalkingCow(),
                ],
            ),
            Entity(
                name="Fat Cow",
                position=Position_2D(1, 0),
                components=[
                    Health(max_health=10, starting_health=10),
                    TalkingCow(),
                ],
            ),
            Entity(
                name="Tiny Cow",
                position=Position_2D(1, 0),
                components=[
                    Health(max_health=1, starting_health=1),
                    TalkingCow(),
                ],
            ),
            Entity(
                name="Wall",
                position=Position_2D(-1, 0),
                tags=["wall"],
                components=[Collidable()],
            ),
            # TODO: for a real pike entity, you likely want an Attack_All_Nearby_Entities action instead of Attack
            Entity(
                name="Spike",
                position=Position_2D(0, -2),
                actions=[Attack(name="Poke", damage_amount=1)],
                tags=["item"],
                components=[Follow_Action_Sequence([(Attack, None)])],
            ),
        ],
        entity_order=entity_definition_order,
    )

    for step in range(exp_steps):
        cur_step_actions = []
        for agent_id, agent in enumerate(env.agents):
            observation = env.observe(agent_id)
            action, info = agent.get_component(Agent_Policy).select_action(observation)
            cur_step_actions.append(action)

        env.step(cur_step_actions)


if __name__ == "__main__":
    run_exp()
