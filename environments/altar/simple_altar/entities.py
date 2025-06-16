from dataclasses import dataclass
from word_play.environment import Entity, Entity_State, Entity_Properties, Environment
import random

# TODO: should likely convert the simple altar to use the version of the altar from common_entities.py

@dataclass(slots=True)
class Altar_Properties(Entity_Properties):
	signal: str

class Altar(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		env.altar_signals.append((self.properties.name, self.properties.signal))


@dataclass(slots=True)
class Random_Altar_Properties(Entity_Properties):
	signals: list[str]

class Random_Altar(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		env.altar_signals.append((self.properties.name, random.sample(self.properties.signals, k=1)))