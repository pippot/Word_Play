from word_play.environment import Action_On_Self, Entity


# TODO: I acknowledge that systems where you can select multiple actions per term (ex., harvest and sanction) creates
#	a non-ideal combinatorial explosion of actions. This be fixed pretty easily by allowing the agent to select multiple
#	actions per turn. However, I have explicity not done this because I believe that the situations where the combinatorial
#	explosion of actions is unmanagable (ex., this alter env is still acceptable) are rare. And I also believe that selecting
#	just a single action per turn makes it easier for llm agents and also simplies creation of new environments. For example,
#	you don't need to worry about stopping the player from selecting two actions which should never happen at the same time.
#	I think this problem is 100% worth revisiting! ...Was not very fun writing out the functions lol


# If you really don't want to write out all the action class combinations, you can create them programmatically.
# See here: https://www.geeksforgeeks.org/create-classes-dynamically-in-python/

# writing out all the actions classes manually:


def get_harvest_sanction_text(action) -> str:
	return f'Harvest {action.harvest} and sanction {action.sanction}.'


class Harvest_Apples_Sanction_Nothing(Action_On_Self):
	harvest='apples'
	sanction='nothing'

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return get_harvest_sanction_text(Harvest_Apples_Sanction_Nothing)
	
	@staticmethod
	def __call__(target_entity, env):
		pass


class Harvest_Bananas_Sanction_Nothing(Action_On_Self):
	harvest='bananas'
	sanction='nothing'

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return get_harvest_sanction_text(Harvest_Bananas_Sanction_Nothing)
	
	@staticmethod
	def __call__(target_entity, env):
		pass


class Harvest_Apples_Sanction_Apples(Action_On_Self):
	harvest='apples'
	sanction='apples'

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return get_harvest_sanction_text(Harvest_Apples_Sanction_Apples)
	
	@staticmethod
	def __call__(target_entity, env):
		pass


class Harvest_Bananas_Sanction_Apples(Action_On_Self):
	harvest='bananas'
	sanction='apples'

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return get_harvest_sanction_text(Harvest_Bananas_Sanction_Apples)
	
	@staticmethod
	def __call__(target_entity, env):
		pass


class Harvest_Apples_Sanction_Bananas(Action_On_Self):
	harvest='apples'
	sanction='bananas'

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return get_harvest_sanction_text(Harvest_Apples_Sanction_Bananas)
	
	@staticmethod
	def __call__(target_entity, env):
		pass


class Harvest_Bananas_Sanction_Bananas(Action_On_Self):
	harvest='bananas'
	sanction='bananas'

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return get_harvest_sanction_text(Harvest_Bananas_Sanction_Bananas)
	
	@staticmethod
	def __call__(target_entity, env):
		pass