from word_play.environment import Action_On_Self, Entity


class Cook_A_Pie(Action_On_Self):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Cook a delicious pie.'
	
	@staticmethod
	def __call__(target_entity, env):
		# this is a dummy function
		pass


class Play_Poker(Action_On_Self):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Play a game of poker with the players around you.'
	
	@staticmethod
	def __call__(target_entity, env):
		# this is a dummy function
		pass
	
class Exercise(Action_On_Self):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Do some exercise to stay fit.'
	
	@staticmethod
	def __call__(target_entity, env):
		# this is a dummy function
		pass