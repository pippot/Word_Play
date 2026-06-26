from __future__ import annotations

from word_play.core import Component, Entity, Environment
from word_play.presets.movement.simple_2d_grid import Position_2D
from word_play.presets.systems.inventory import Inventory
from word_play.presets.systems.reward import award_reward

from benchmarks.text_mp.core.timing import BENCHMARK_STEPS


CHEMISTRY_START_POSITIONS = [
    (9, 9),
    (10, 9),
    (11, 9),
    (12, 9),
    (13, 9),
    (14, 9),
    (15, 9),
    (8, 8),
]

MOLECULE_NAMES = {
    "energy": "Energy",
    "x": "Molecule X",
    "y": "Molecule Y",
    "ix": "Inhibitor IX",
    "iy": "Inhibitor IY",
    "food1": "Food 1",
    "food2": "Food 2",
    "food3": "Food 3",
    "distractor": "Distractor",
}


class Molecule(Component):
    def __init__(self, kind: str, holding_reward: float = 0.0):
        super().__init__(tags=["item", "molecule", kind])
        self.kind = kind
        self.holding_reward = holding_reward

    def pre_actions_step(self, env: Environment) -> None:
        chemistry_events(env)
        if carrier_of(env, self.entity) is not None:
            return
        for agent in env.agents:
            inventory = agent.get_component(Inventory)
            if inventory is None or inventory.contents:
                continue
            if abs(agent.position.x - self.entity.position.x) + abs(agent.position.y - self.entity.position.y) <= 1:
                if inventory.store(self.entity, env):
                    chemistry_events(env).append(f"{agent.name} picked up {self.entity.name}.")
                return

    def post_actions_step(self, env: Environment) -> None:
        if self.entity not in env.state.entities:
            return
        maybe_end_chemistry_episode(env)
        if self.kind in {"food1", "food2", "food3"}:
            self._metabolize_if_carried(env)
        elif self.kind == "distractor":
            self._reward_if_carried(env)
        elif self.kind == "x":
            self._combine_with_y(env)

    def _metabolize_if_carried(self, env: Environment) -> None:
        carrier = carrier_of(env, self.entity)
        if carrier is None:
            return
        reward = 10.0 if self.kind == "food3" else 1.0
        remove_molecule(env, self.entity)
        award_reward(env, carrier, reward)
        chemistry_events(env).append(f"{carrier.name} metabolized {self.entity.name} for +{reward:g}.")

    def _reward_if_carried(self, env: Environment) -> None:
        if self.holding_reward <= 0:
            return
        carrier = carrier_of(env, self.entity)
        if carrier is None:
            return
        award_reward(env, carrier, self.holding_reward)
        chemistry_events(env).append(f"{carrier.name} held {self.entity.name} for +{self.holding_reward:g}.")

    def _combine_with_y(self, env: Environment) -> None:
        position = molecule_position(self.entity, env)
        y_molecule = first_molecule_near(env, "y", position, exclude=self.entity)
        if y_molecule is None:
            return

        carriers = []
        for carrier in (carrier_of(env, self.entity), carrier_of(env, y_molecule)):
            if carrier is not None and carrier not in carriers:
                carriers.append(carrier)

        remove_molecule(env, self.entity)
        remove_molecule(env, y_molecule)
        env.instantiate_entity(molecule_entity("energy", position))

        for carrier in carriers:
            award_reward(env, carrier, 10.0)

        reward_text = ", ".join(f"{carrier.name} +10" for carrier in carriers) or "no carried molecule reward"
        chemistry_events(env).append(f"MetabolizeXY at {position}: x + y -> energy ({reward_text}).")


class CycleSite(Component):
    def __init__(self, cycle: str):
        super().__init__(tags=["cycle_site", f"{cycle}_cycle"])
        self.cycle = cycle

    def pre_actions_step(self, env: Environment) -> None:
        chemistry_events(env)

    def post_actions_step(self, env: Environment) -> None:
        if self.entity not in env.state.entities:
            return
        maybe_end_chemistry_episode(env)
        energy = first_molecule_near(env, "energy", self.entity.position)
        if energy is None:
            return

        energy_carrier = carrier_of(env, energy)
        reaction_position = molecule_position(energy, env)
        remove_molecule(env, energy)
        products = self.products(env)
        for product in products:
            env.instantiate_entity(molecule_entity(product, reaction_position))

        carrier_text = f" using {energy_carrier.name}'s carried energy" if energy_carrier else ""
        chemistry_events(env).append(
            f"{self.cycle.title()} cycle fired near {self.entity.position}{carrier_text}: "
            f"energy -> {', '.join(products)}."
        )

    def products(self, env: Environment) -> list[str]:
        products = (
            ["food1", "x", "iy"] if self.cycle == "blue" else
            ["food2", "y", "ix"] if self.cycle == "green" else
            ["food3", "food1", "food2"]
        )
        inhibitor = "iy" if self.cycle == "blue" else "ix" if self.cycle == "green" else None
        blocked_product = "x" if self.cycle == "blue" else "y" if self.cycle == "green" else None
        if inhibitor is not None and first_molecule_near(env, inhibitor, self.entity.position, distance=0) is not None:
            products.remove(blocked_product)
            chemistry_events(env).append(
                f"{self.cycle.title()} cycle inhibited at {self.entity.position}: "
                f"{inhibitor} blocked {blocked_product}."
            )
        return products


def molecule_entity(kind: str, position: Position_2D) -> Entity:
    return Entity(
        name=MOLECULE_NAMES[kind],
        position=Position_2D(position.x, position.y),
        components=[Molecule(kind)],
    )


def chemistry_events(env: Environment) -> list[str]:
    if getattr(env, "_chemistry_events_step", None) != env.cur_step:
        env.tick = env.cur_step
        env.chemistry_events = []
        env._chemistry_events_step = env.cur_step
    return env.chemistry_events


def maybe_end_chemistry_episode(env: Environment) -> None:
    if getattr(env, "_chemistry_end_checked_step", None) == env.cur_step:
        return
    env._chemistry_end_checked_step = env.cur_step
    max_steps = getattr(env, "max_episode_steps", BENCHMARK_STEPS)
    if env.cur_step + 1 >= max_steps:
        env.truncations = [True for _ in env.agents]


def molecule_kind(entity: Entity) -> str | None:
    molecule = entity.get_component(Molecule)
    return molecule.kind if molecule is not None else None


def carrier_of(env: Environment, item: Entity) -> Entity | None:
    for agent in env.agents:
        inventory = agent.get_component(Inventory)
        if inventory is not None and item in inventory.contents:
            return agent
    return None


def molecule_position(entity: Entity, env: Environment) -> Position_2D:
    carrier = carrier_of(env, entity)
    return carrier.position if carrier is not None else entity.position


def first_molecule_near(
    env: Environment,
    kind: str,
    position: Position_2D,
    exclude: Entity | None = None,
    distance: int = 1,
) -> Entity | None:
    for entity in env.state.entities:
        entity_position = molecule_position(entity, env)
        if (
            entity is not exclude
            and molecule_kind(entity) == kind
            and abs(entity_position.x - position.x) + abs(entity_position.y - position.y) <= distance
        ):
            return entity
    return None


def remove_molecule(env: Environment, molecule: Entity) -> None:
    carrier = carrier_of(env, molecule)
    if carrier is not None:
        inventory = carrier.get_component(Inventory)
        if inventory is not None and molecule in inventory.contents:
            inventory.contents.remove(molecule)
    while "in_inventory" in molecule.tags:
        molecule.tags.remove("in_inventory")
    if molecule in env.state.entities:
        env.destroy_entity(molecule)
