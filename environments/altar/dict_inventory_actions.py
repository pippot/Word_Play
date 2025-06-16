from word_play.environment import Action_On_Other_Entity, Entity, Environment


class Pick_Fruit(Action_On_Other_Entity):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Pick a {target_entity.properties.fruit}.'

	@staticmethod
	def __call__(target_entity: Entity, actor: Entity, env: Environment):
		fruit = target_entity.properties.fruit
		if fruit not in actor.state.inventory:
			actor.state.inventory[fruit] = 0
		actor.state.inventory[fruit] += 1