from dataclasses import dataclass
from word_play.environment import Entity, Entity_State, Entity_Properties, Environment
import random
from environments.altar.common_utility import CLAN_NAME
from environments.altar.common_actions import Pick_Fruit
from typing import NamedTuple
from enum import Enum


# TODO: right now the altar is providing a global signal, we might want to change this.
# 		Also, if we are providing a global signal, it would likely just be easier to set the signal in the environment...

class Altar_Signal_Type(Enum):
	APPLE, BANANA, PEACH, ORANGE, PLUM = range(5)


ALTER_SIGNAL_TO_MESSAGE = {
	Altar_Signal_Type.APPLE: f"Valued citizens of {CLAN_NAME}, let's focus on harvesting apples. It is important for the prosperity of our community!",
	Altar_Signal_Type.BANANA: f"Valued citizens of {CLAN_NAME}, let's focus on harvesting bananas. It is important for the prosperity of our community!",
	Altar_Signal_Type.PEACH: f"Valued citizens of {CLAN_NAME}, let's focus on harvesting peaches. It is important for the prosperity of our community!",
	Altar_Signal_Type.ORANGE: f"Valued citizens of {CLAN_NAME}, let's focus on harvesting oranges. It is important for the prosperity of our community!",
	Altar_Signal_Type.PLUM: f"Valued citizens of {CLAN_NAME}, let's focus on harvesting plums. It is important for the prosperity of our community!",
}


class Altar_Signal(NamedTuple):
	altar_name: str
	signal: Altar_Signal_Type
	signal_message: str


@dataclass(slots=True)
class Altar_Properties(Entity_Properties):
	signal: Altar_Signal_Type

class Altar(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		env.altar_signals.append(Altar_Signal(
									altar_name=self.properties.name,
									signal=self.properties.signal,
									signal_message=ALTER_SIGNAL_TO_MESSAGE[self.properties.signal]
									))


@dataclass(slots=True)
class Random_Altar_Properties(Entity_Properties):
	signals: list[Altar_Signal_Type]

class Random_Altar(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		env.altar_signals.append((self.properties.name, random.sample(self.properties.signals, k=1)))


@dataclass(slots=True)
class Fruit_Tree_Properties(Entity_Properties):
	fruit: str

class Fruit_Tree(Entity):

	exposed_actions = (Pick_Fruit(),)
	
	def __init__(self, state: Entity_State, properties: Fruit_Tree_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		pass