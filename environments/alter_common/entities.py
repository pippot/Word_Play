from dataclasses import dataclass
from word_play.environment import Entity, Entity_State, Entity_Properties, Environment
import random


@dataclass(slots=True)
class Altar_Properties(Entity_Properties):
	signal: str

class Altar(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		env.alter_signals.append((self.properties.name, self.properties.signal))


@dataclass(slots=True)
class Random_Altar_Properties(Entity_Properties):
	signals: list[str]

class Random_Alter(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		env.alter_signals.append((self.properties.name, random.sample(self.properties.signals, k=1)))