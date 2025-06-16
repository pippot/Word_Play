from word_play.environment import Action_On_Other_Entity, Entity


class PickBerry(Action_On_Other_Entity):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Pick a {target_entity.properties.berry_type} berry from {target_entity.properties.name}.'
	
	@staticmethod
	def __call__(target_entity, actor, env):
		print(f'|| (target_entity: {target_entity}, actor: {actor}, env: {env}) Action: {PickBerry.action_description_text(target_entity)} ||')


class Unlock(Action_On_Other_Entity):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Unlock {target_entity.properties.name}.'
	
	@staticmethod
	def __call__(target_entity, actor, env):
		print(f'|| (target_entity: {target_entity}, actor: {actor}, env: {env}) Action: {Unlock.action_description_text(target_entity)} ||')
