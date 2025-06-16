from word_play.environment import Observation, Agent, Action_Selection, Action, Entity, Entity_State, Entity_Properties
from word_play.model import Model
from word_play.presets.string_utils import extract_number_surrounded_by_quotes, remove_potential_leading_and_trailing_quotes
import random
import re
import traceback
from copy import deepcopy


class Random_Action_Agent(Agent):
	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		return observation.possible_actions[random.randrange(len(observation.possible_actions))], {}

	def step(self):
		pass


class Constant_Strategy_Agent(Agent):
	def __init__(self, state: Entity_State, properties: Entity_Properties, constant_action: Action, constant_action_target: Entity | str) -> None:
		super().__init__(state=state, properties=properties)
		self.constant_action_selection = Action_Selection(
			action=constant_action,
			target_entity=self if constant_action_target == 'self' else constant_action_target
		)
	
	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		return self.constant_action_selection, {}

	def step(self):
		pass


# TODO: might be nice to have retrying as a decorator or some kind of wrapper function


class Explicit_Belief_Agent(Agent):

	def __init__(
			self,
			state: Entity_State,
			properties: Entity_Properties,
			model: Model,
			env_description: str,
			history_length: int=3,
			model_gen_retry_count: int=5) -> None:
		'''If history_length is <0, we display the full history.'''
		super().__init__(state=state, properties=properties)
		self.model = model
		self.env_description = env_description
		self.history_length = history_length
		self.model_gen_retry_count = model_gen_retry_count
		self.obs_history = []
		self.my_action_history = ['None']
		self.expectations = 'None'	# for now, we'll just let the model structure this as it pleases


	def get_history_enumerator(self, obs_history, my_action_history):
		if self.history_length > 0:
			return enumerate(
				zip(obs_history[-self.history_length:], my_action_history[-self.history_length:]),
				start=max(0, len(my_action_history) - self.history_length))
		elif self.history_length == 0:
			return enumerate([])
		elif self.history_length < 0:
			return enumerate(zip(obs_history, my_action_history))
			

	def history_to_str(self, obs_history, my_action_history) -> str:
		history_str = f''
		# iterate over the last history_length rounds
		for round, (obs, my_action) in self.get_history_enumerator(obs_history, my_action_history):
			history_str += f'\n\n## Round {round} Observation:\n\n{obs}\n\n## Round {round} My Action: {my_action}'
		if not history_str:
			history_str = 'No prior history.'

		return history_str.strip()


	def update_expectations(self, obs_history, my_action_history) -> str:
		prompt = f"""# Game Description:
{self.env_description}

# History:
{self.history_to_str(obs_history, my_action_history)}

# Current Expectations:
{self.expectations}

# New Expectations:
New expectation replace all current expections. How do you expect other players will act? Be thorough and explicit. You have a 280 character limit."""
		
		generated_text = self.model.generate_text(input_text=prompt).strip()
		if generated_text.startswith('# New Expectations:'):
			generated_text = generated_text[len('# New Expectations:'):].strip()
		self.expectations = generated_text

		return self.expectations, prompt
	

	def ask_model_to_select_action(self, obs_history, my_action_history) -> tuple[int, str]:
		prompt = f"""# Game Description:
{self.env_description}

# History:
{self.history_to_str(obs_history, my_action_history)}

# Expectations:
{self.expectations}

# Current Round:
What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, output the index of your final action selection surrounded by quotation marks (e.g. Final Answer: "<action_selection>"):"""
		
		selection_CoT = self.model.generate_text(prompt).strip()
		action_idx = extract_number_surrounded_by_quotes(selection_CoT)
		return action_idx, selection_CoT, prompt


	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		# update histories
		self.obs_history.append(observation)

		new_expectations, new_expectations_prompt = self.update_expectations(self.obs_history, self.my_action_history)
		successfully_selected_action = False
		for _ in range(self.model_gen_retry_count):
			try:
				action_idx, selection_CoT, CoT_prompt = self.ask_model_to_select_action(
																self.obs_history,
																self.my_action_history
															)
				selected_action = observation.possible_actions[action_idx]
				successfully_selected_action = True
				break
			except ValueError:
				pass
			except IndexError:
				pass
		if not successfully_selected_action:
			raise Exception(f'Failed to select action after {self.model_gen_retry_count} attempts.\nselection_CoT: {selection_CoT}\nCoT_prompt: {CoT_prompt}.')

		self.my_action_history.append(selected_action)
		return selected_action, {
			'new_expectations': new_expectations,
			'selection_CoT': selection_CoT,
			'new_expectations_prompt': new_expectations_prompt,
			'CoT_prompt': CoT_prompt,
		}


	def step(self):
		pass


class Explicit_Belielf_Agent_With_Simple_Conversation(Explicit_Belief_Agent):

	def __init__(
			self,
			state: Entity_State,
			properties: Entity_Properties,
			model: Model,
			env_description: str,
			history_length: int=3,
			model_gen_retry_count: int=5,
			#max_conversation_history_len: int=3
			) -> None:
		super().__init__(state=state, properties=properties, model=model, env_description=env_description, history_length=history_length, model_gen_retry_count=model_gen_retry_count)
		#self.max_conversation_history_len = max_conversation_history_len
		#self.conversation_history = []


	def extract_speech_and_action_selection(self, input_str):
		texts_surrounded_by_quotes = re.findall('"[^"]+"', input_str)
		action_idx = int(texts_surrounded_by_quotes[-1][1:-1])
		speech_message = texts_surrounded_by_quotes[-2][1:-1]
		return action_idx, speech_message


	def ask_model_to_select_action(self, obs_history, my_action_history) -> tuple[int, str]:
		prompt = f"""# Game Description:
{self.env_description}

# History:
{self.history_to_str(obs_history, my_action_history)}

# Expectations:
{self.expectations}

# Current Round:
You may both select an action and say something to the players around you.
What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, first output the message you would like to send to nearby players and then output the index of your final action selection surrounded by quotation marks.
If you don't wish to say anything to other players, simply output an empty string.

Example output format:
'This is my thought process...
Message To Other Player: "<my_message>"
Final Answer: "<action_selection>".'"""
		
		selection_CoT = self.model.generate_text(prompt).strip()
		action_idx, speech_message = self.extract_speech_and_action_selection(selection_CoT)
		return action_idx, speech_message, selection_CoT, prompt


	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		# update histories
		self.obs_history.append(observation)
		# TODO: we likely want to add more concrete support for conversations
		#self.conversation_history.append(observation.conversation)
		#self.conversation_history = self.conversation_history[-self.max_conversation_history_len:]

		new_expectations, new_expectations_prompt = self.update_expectations(self.obs_history, self.my_action_history)
		successfully_selected_action = False
		last_error_message = None
		for _ in range(self.model_gen_retry_count):
			try:
				action_idx, speech_message, selection_CoT, CoT_prompt = self.ask_model_to_select_action(
																			self.obs_history,
																			self.my_action_history
																		)
				selected_action = observation.possible_actions[action_idx]
				successfully_selected_action = True
				break
			except ValueError:
				last_error_message = traceback.format_exc()
			except IndexError:
				last_error_message = traceback.format_exc()
		if not successfully_selected_action:
			raise Exception(f'ERROR:\n{last_error_message}\n\nFailed to select action after {self.model_gen_retry_count} attempts.')
		
		self.my_action_history.append(selected_action)
		return selected_action, {
			'speech_message': speech_message,
			'new_expectations': new_expectations,
			'selection_CoT': selection_CoT,
			'new_expectations_prompt': new_expectations_prompt,
			'CoT_prompt': CoT_prompt,
		}


# TODO: this class need to be rewritten to be simpler!
class Explicit_Belief_Agent_With_Discussion_Phase(Explicit_Belief_Agent):

	def ask_model_to_create_discussion_message(self, obs_history, my_action_history) -> str:
		prompt = f"""# Game Description:
{self.env_description}

# History:
{self.history_to_str(obs_history, my_action_history)}

# Expectations:
{self.expectations}

# Current Discussion Phase 
What do you want to say (You have a 280 character limit)?"""
		
		message = self.model.generate_text(input_text=prompt).strip()
		message	= remove_potential_leading_and_trailing_quotes(message)
		return message, prompt


	def get_discussion_message(self, observation: Observation) -> tuple[str, dict]:
		# the current observation is the same as the most recent observation with the exception of the discussion messages.
		# we will temporarily add the current observation history
		# TODO: we should avoid deepcopy
		temp_obs_history = deepcopy(self.obs_history)
		temp_obs_history.append(observation)

		new_expectations, new_expectations_prompt = self.update_expectations(temp_obs_history, self.my_action_history)
		discussion_message, discussion_message_prompt = self.ask_model_to_create_discussion_message(
																temp_obs_history,
																self.my_action_history
															)
		
		return discussion_message, {
			'new_expectations': new_expectations,
			'new_expectations_prompt': new_expectations_prompt,
			'discussion_message_prompt': discussion_message_prompt,
		}
	

	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		# this is sketchy hotfix to make it so that the most recent action is None instead of the first action
		# we will acheive this by simulating the actions always being added at the second last index
  		# (right before the initially generated 'None' action)
		# TODO: his will be rewritten to be simpler
		latest_action, info = super().select_action(observation)
		self.my_action_history.pop(-1)
		self.my_action_history.insert(-1, latest_action)
		return latest_action, info
