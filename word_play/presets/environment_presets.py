from word_play.environment import Environment, Environment_State, Environment_Properties, Movement_System, Action_Selection, Step_Execution_Order
from typing import Callable
import copy
from typing import NamedTuple 


# TODO: would be really nice if we have a factory or something which is able to combine multiple Environment presets together
#	I think such a (potentially) factory would need to be manually created, since there is no guarentee that different presets
#	will not conflict with each other


# TODO: this is not terribly effecient
class Simple_Reset_Environment(Environment):

	def __init__(
			self,
			state: Environment_State,
			properties: Environment_Properties,
			movement_system: Movement_System,
			reward_func: Callable[[list[Action_Selection, Environment]], list[float]],
			step_execution_order: Step_Execution_Order = None
		) -> None:
		self.initial_state = copy.deepcopy(state)
		init_kwargs = {}
		if step_execution_order is not None:
			init_kwargs['step_execution_order'] = step_execution_order
		super().__init__(state=state, properties=properties, movement_system=movement_system, reward_func=reward_func, **init_kwargs)

	def _reset(self, seed=None) -> None:
		self.state = copy.deepcopy(self.initial_state)


# TODO: should name this
class Conversation_And_Reset_Environment(Environment):

	def __init__(
			self,
			state: Environment_State,
			properties: Environment_Properties,
			movement_system: Movement_System,
			reward_func: Callable[[list[Action_Selection, Environment]], list[float]],
			step_execution_order: Step_Execution_Order = None
		) -> None:
		self.initial_state = copy.deepcopy(state)
		init_kwargs = {}
		if step_execution_order is not None:
			init_kwargs['step_execution_order'] = step_execution_order
		
		super().__init__(state=state, properties=properties, movement_system=movement_system, reward_func=reward_func, **init_kwargs)
		
		self.conversation = ['' for _ in range(len(self.agents))]

	def _reset(self, seed=None) -> None:
		self.state = self.initial_state
	
	def step(self, action_selections: list[Action_Selection], conversation: list[str]) -> None:
		assert len(conversation) == len(self.agents)
		self.conversation = conversation
		return super().step(action_selections)


class Message(NamedTuple):
	sender_id: int
	message: str


class Discussion_Phase_With_Reset_Environment(Environment):

	def __init__(
			self,
			state: Environment_State,
			properties: Environment_Properties,
			movement_system: Movement_System,
			reward_func: Callable[[list[Action_Selection, Environment]], list[float]],
			step_execution_order: Step_Execution_Order = None,
			discussion_phase_turn_count: int = 3
		) -> None:
		self.initial_state = copy.deepcopy(state)
		self.discussion_phase_turn_count = discussion_phase_turn_count
		self.in_discussion_phase = True
		# NOTE: we may safely store conversations in a single list since no actions are taken during the discussion phase
		# thus we don't need to worry about agents no longer being close to each other (or similar complications).
		self.cur_discussion_messages: list[list[Message]] = [[]]

		init_kwargs = {}
		if step_execution_order is not None:
			init_kwargs['step_execution_order'] = step_execution_order
		super().__init__(state=state, properties=properties, movement_system=movement_system, reward_func=reward_func, **init_kwargs)


	def _reset(self, seed=None) -> None:
		self.state = copy.deepcopy(self.initial_state)
	

	def submit_message(self, sender_id: int, message: str) -> None:
		# NOTE: could have "recipient_id" (or recipient_ids) input_parameter to support both public and private messages
		self.cur_discussion_messages[-1].append(Message(sender_id, message))


	def end_discussion_phase_turn(self) -> None:
		'''
		We decide to have this method control when discussion phase turns end in case the user wants to a have some agents not submit any messages.
		'''
		if len(self.cur_discussion_messages) < self.discussion_phase_turn_count:
			self.cur_discussion_messages.append([])


	def end_discussion_phase(self) -> None:
		'''
		We decide to have this method control when the discussion phase ends in case the user wants to end the discussion phase early.
		This type of method would like also be useful if you would like your environment to have a an indeterminate number of discussion phase turns.
		'''
		self.in_discussion_phase = False


	def start_new_discussion_phase(self) -> None:
		self.in_discussion_phase = True	# NOTE: not sure if we should set in_discussion_phase to True here or after step completes
		self.cur_discussion_messages = [[]]

	
	def step(self, action_selections: list[Action_Selection]) -> None:
		assert self.in_discussion_phase == False, 'The discussion phase must complete before the action/step phase.\
Use the conclude_discussion_phase method to end the discussion phase.'

		super().step(action_selections)