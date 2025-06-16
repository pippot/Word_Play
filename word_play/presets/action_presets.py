from word_play.environment import Action_On_Self, Entity


class Do_Nothing(Action_On_Self):

	@staticmethod
	def action_description_text(target_entity: Entity) -> str:
		return f'Do nothing.'
	
	@staticmethod
	def __call__(target_entity, env):
		return 'You did nothing.'



# # TODO: this is just a temp example
# def pick_berry(self, actor, world_state):
# 	self.properties.berry_count -= 1
# 	actor.properties.berry_count += 1

# def pick_up_item(self, actor, world_state):
# 	actor.inventory.append(self)

# def unlock(self, actor, world_state):
# 	if self.required_key in actor.inventory:
# 		self.locked = False

# def sleep(target_entity, env):
# 	target_entity.rest_amount += 3
