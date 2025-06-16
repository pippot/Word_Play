from dataclasses import dataclass
from word_play.environment import Entity, Entity_State, Entity_Properties, Environment
from environments.altar.dict_inventory_actions import Pick_Fruit


@dataclass(slots=True)
class Infinite_Fruit_Tree_Properties(Entity_Properties):
	fruit: str

class Infinite_Fruit_Tree(Entity):

	exposed_actions = (Pick_Fruit,)

	def __init__(self, state: Entity_State, properties: Infinite_Fruit_Tree_Properties) -> None:
		super().__init__(state=state, properties=properties)

	def step(self, env: Environment):
		pass