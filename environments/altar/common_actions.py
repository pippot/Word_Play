from word_play.environment import Action_On_Other_Entity, Entity, Environment


class Pick_Fruit(Action_On_Other_Entity):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Harvest {target_entity.properties.fruit} from {target_entity.properties.name}'
	
	@staticmethod
	def __call__(target_entity: Entity, actor: Entity, env: Environment):
		pass