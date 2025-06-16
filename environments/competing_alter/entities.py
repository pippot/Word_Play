from dataclasses import dataclass
from word_play.environment import Entity, Entity_State, Entity_Properties, Environment, Action_On_Self
import random
from environments.alter_common.actions import get_harvest_sanction_text

@dataclass(slots=True)
class Altar_Properties(Entity_Properties):
	signal: Action_On_Self

class Altar(Entity):

	exposed_actions = ()

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)
	
	def get_signal(self):
		return self.properties.signal

	def step(self, env: Environment):
		env.alter_signals.append((self.properties.name, get_harvest_sanction_text(self.properties.signal)))


@dataclass(slots=True)
class Random_Altar_Properties(Entity_Properties):
	signals: list[Action_On_Self]

class Random_Alter(Entity):

	exposed_actions = ()
	current_signal = None

	def __init__(self, state: Entity_State, properties: Altar_Properties) -> None:
		super().__init__(state=state, properties=properties)
		self.get_random_action()
	
	def get_random_action(self):
		self.current_signal = random.sample(self.properties.signals, k=1)[0]

	def get_signal(self):
		return self.current_signal

	def step(self, env: Environment):
		self.get_random_action()
		env.alter_signals.append((self.properties.name, get_harvest_sanction_text(self.current_signal)))
	