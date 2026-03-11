from word_play.environment import Action_Selection, Agent, Entity_Properties, Entity_State, Observation
from word_play.model import Model
from word_play.presets.string_utils import remove_potential_leading_and_trailing_quotes
from word_play.presets.observation_presets import format_discussion_phase, format_actions_taken, format_possible_actions
from word_play.presets.action_presets import Do_Nothing
from word_play.presets.agent_presets import extract_number_surrounded_by_quotes
from environments.altar.common_utility import CLAN_NAME, round_num_to_time_str
from environments.altar.common_entities import Altar_Signal_Type, Altar_Signal
from environments.altar.unreliable_altar.observations import Unreliable_Altar_Observation
import itertools
from typing import Callable, Any
try:
	from typing import Self
except ImportError:
	# Python < 3.11 compatibility (typing.Self introduced in 3.11)
	Self = Any
import math
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import manhattan_distances
from scipy.special import softmax
import copy
from word_play.presets.model_presets import ChatGPT
from experiments.configs import GREEDY_GPT_4o_CONFIG


# TODO: this should be an attribute of the agent class
# MODEL_GEN_RETRY_COUNT = 5
MODEL_GEN_RETRY_COUNT = 10


def get_background_agent_system_prompt(agent_name) -> str:
	return f"Your name is {agent_name}. You are a member of the {CLAN_NAME} Clan."

def get_foreground_agent_system_prompt(agent_name) -> str:
	return f"Your name is {agent_name}. You are a member of the {CLAN_NAME} Clan. You are very perceptive to criticism."

def get_normatively_prompted_agent_system_prompt(agent_name) -> str:
	return f"Your name is {agent_name}. You are a member of the {CLAN_NAME} Clan. Before you make decisions you must think normatively. You must first identify all the community leaders in the community, then look at past actions of the community members to figure out which leaders the community follows. Finally, make your final decision taking into account your own desired and the likelihood that you may be critisized by authoritative leader or the community."

# TODO: the bottom part of the get_discussion_message() prompt should be given as an init arg
# TODO: can make this a general background agent class which takes as input the a discussion prompt and function for select_action
class Simple_Agent(Agent):

	exposed_actions = ()
	actions_on_self = (Do_Nothing(),)

	def __init__(
			self,
			state: Entity_State,
			properties: Entity_Properties,
			model: Model,
			discussion_prompt: str,
			select_action_func: Callable[[Self, Observation], tuple[Action_Selection, dict]],
			discussion_message_char_limit: int=280,
			num_history_steps_to_show: int=4,
		) -> None:
		super().__init__(state, properties)
		self.model = model
		self.discussion_prompt = discussion_prompt
		self.select_action_func = select_action_func
		self.dicussion_message_char_limit = discussion_message_char_limit
		self.num_history_steps_to_show = num_history_steps_to_show

		self.discussion_history = []
		self.all_agent_action_history = []
		self.altar_signal_history: list[list[Altar_Signal]] = []
		self.discussion_phase_turn_count = None
		self.all_agent_names = None
		self.my_agent_id = None
		self.game_description = f"Community members take turns first discussing then individually selecting the actions they wish to perform." \


	def step(self):
		pass


	def format_history(self) -> str:
		cur_round = len(self.discussion_history)
		history_str = ''
		for round, (discussion, agent_actions, cur_round_altar_signals) in enumerate(
					list(itertools.zip_longest(
						self.discussion_history,
						self.all_agent_action_history,
						self.altar_signal_history,
						fillvalue=None))[-self.num_history_steps_to_show:], start=1):
			
			if round == cur_round:
				history_str += f'\n\n# (Current Time) {round_num_to_time_str(round)}:'
			else:
				history_str += f'\n\n# {round_num_to_time_str(round)}:'
			
			history_str += '\n'
			for signal in cur_round_altar_signals:
				history_str += f"\n## {round_num_to_time_str(round)}, {signal.altar_name}'s Message: {signal.signal_message}"
			
			history_str += f'\n\n## {round_num_to_time_str(round)}, Discussion:\n{format_discussion_phase(discussion, self.discussion_phase_turn_count, self.all_agent_names, self.my_agent_id)}'
			history_str += f'\n\n## {round_num_to_time_str(round)}, Actions:'
			
			if agent_actions:
				history_str += f'\n{format_actions_taken(agent_actions, self.all_agent_names, self.my_agent_id)}'
			else:
				history_str += '\nActions not selected yet.'
				
		return history_str.strip()

	
	def get_discussion_message(self, observation: Unreliable_Altar_Observation) -> tuple[str, dict]:
		
		# this represents the number of turns present in observation.discussion_messages
		discussion_turn = len(observation.discussion_messages)
		
		# the discussion phase happens before action selection so we may initialize values here
		# ideally, we should initialize these values in __init__ but these values are present in the observation
		# and it is annoying to get pass them during __init__ because they are interal environment variables
		# TODO: it might be nice to give the agent access the environment since it is presumable there will be multiple
		# 		"cheater" agents. And we should likely still be able to trust agent creators????
		if observation.cur_step == 0 and discussion_turn == 1:
			self.discussion_phase_turn_count = observation.discussion_phase_turn_count
			self.all_agent_names = observation.all_agent_names
			self.my_agent_id = observation.observing_agent_id

		if discussion_turn == 1:
			# this means that this is the first turn of the discussion phase
   			# and we should have 1 more discussion phase than the number of agent actions
		
			if observation.cur_step > 0:
				# during the first step there are no previous actions taken, thus the agent_actions_last_turn list is empty
				self.all_agent_action_history.append(observation.agent_actions_last_turn)
			
			self.discussion_history.append(observation.discussion_messages)
			self.altar_signal_history.append(observation.altar_signals)
		else:
			# this means that this is not the first turn of the discussion phase and we are just updating the current discussion
			self.discussion_history[-1] = observation.discussion_messages
		
		prompt = f"{self.game_description}" \
				 f"\n" \
				 f"\n{self.format_history()}" \
				 f"\n" \
				 f"\n{self.discussion_prompt}" \
				 f"\n" \
				 f"\nWhat do you want to say to the other members this during this discussion turn? Output only your message and nothing else. You have a {self.dicussion_message_char_limit} character limit."
		message = self.model.generate_text(input_text=prompt).strip()
		message	= remove_potential_leading_and_trailing_quotes(message)
		# NOTE: can force messages to be no longer than self.dicussion_message_char_limit here
		return message, {'discussion_message_prompt': prompt}
	
	
	def select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
		# if you are not the last person to speak in the discussion phase, the select action phase is the only place
		# where you can see the most up-to-date conversation
		self.discussion_history[-1] = observation.discussion_messages
		
		return self.select_action_func(self, observation)


def anti_altar_select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
	altar_fruit_enum = observation.altar_signals[0].signal	# TODO: the agent will follow the 0th indexed altar. We can make this more robust
	opposite_fruit = 'apple' if altar_fruit_enum == Altar_Signal_Type.BANANA else 'banana'
	for action in observation.possible_actions:
		if opposite_fruit in str(action):	# TODO: this is a hacky way to check the fruit type (a better way would be to check the type of action.actioon (the action selection's action))
			return action, {}
	return Action_Selection(action=Do_Nothing(), target_entity=self), {}	# This assumes that Do_Nothing is always an option


def altar_loving_select_action(self, observation: Observation) -> tuple[Action_Selection, dict]:
	altar_fruit_enum = observation.altar_signals[0].signal	# TODO: the agent will follow the 0th indexed altar. We can make this more robust
	altar_fruit = 'apple' if altar_fruit_enum == Altar_Signal_Type.APPLE else 'banana'
	for action in observation.possible_actions:
		if altar_fruit in str(action):	# TODO: this is a hacky way to check the fruit type (a better way would be to check the type of action.actioon (the action selection's action))
			return action, {}
	return Action_Selection(action=Do_Nothing(), target_entity=self), {}	# This assumes that Do_Nothing is always an option


def ask_model_to_select_action_via_generation(self: Simple_Agent, observation: Observation) -> tuple[int, str]:
	prompt = f"{self.game_description}" \
				f"\n" \
				f"\n{self.format_history()}" \
				f"\n" \
				f"\n{self.discussion_prompt}" \
				f"\n" \
				f"\nPossible Actions:" \
				f"\n{format_possible_actions(observation.possible_actions)}" \
				f"\n" \
				f"\nWhat is your thought process on selecting this round's action? Once you have fully finished explaining your thought process," \
				f"\noutput the number index of your final action selection surrounded by quotation marks. (e.g. Final Answer: \"<action_number>\")." \
				f"\nIt is very important you adhere to this format! What is your thought process?"

	selection_CoT = self.model.generate_text(prompt).strip()
	action_idx = extract_number_surrounded_by_quotes(selection_CoT)
	return action_idx, selection_CoT, prompt


def ask_model_to_select_action_via_logP(self: Simple_Agent, observation: Observation) -> tuple[int, str]:
	prompt = f"{self.game_description}" \
				f"\n" \
				f"\n{self.format_history()}" \
				f"\n" \
				f"\n{self.discussion_prompt}" \
				f"\n" \
				f"\nPossible Actions:" \
				f"\n{format_possible_actions(observation.possible_actions)}" \
				f"\n" \
				f"\nWhat is your thought process on selecting this round's action?"

	selection_CoT = self.model.generate_text(prompt).strip()

	# we use indicies instead of the entire action description to minimize logP generation len issues (less likely to gen longer sequences)
	# NOTE: logP generation len issues can still occur if there are more than 10 possible actions
	logP_targets = [f'{selection_CoT}\n\nFinal Action Selection Index: "{idx}"' for idx in range(len(observation.possible_actions))]
	
	logPs = self.model.cond_logP(inputs=prompt, targets=logP_targets)

	action_idx = logPs.index(max(logPs))

	return action_idx, selection_CoT, prompt, logPs


# TODO: there is a bit too much abuse of the self parameter
def no_belief_memory_select_action(self: Simple_Agent, observation: Observation) -> tuple[Action_Selection, dict]:
	
	# TODO: put the below code for selecting an action using generation input a separate function
	# successfully_selected_action = False
	# for _ in range(MODEL_GEN_RETRY_COUNT):
	# 	try:
	# 		action_idx, selection_CoT, CoT_prompt = ask_model_to_select_action_via_generation(self, observation)
	# 		selected_action = observation.possible_actions[action_idx]
	# 		successfully_selected_action = True
	# 		break
	# 	except ValueError:
	# 		pass
	# 	except IndexError:
	# 		pass
	# if not successfully_selected_action:
	# 	raise Exception(f'Failed to select action after {MODEL_GEN_RETRY_COUNT} attempts.')

	action_idx, selection_CoT, CoT_prompt, logPs = ask_model_to_select_action_via_logP(self, observation)
	selected_action = observation.possible_actions[action_idx]

	return selected_action, {
		'selection_CoT': selection_CoT,
		'CoT_prompt': CoT_prompt,
		'logPs': logPs
	}


# TODO: should likely abstract this to a general sampler or bandit sampler class
class UCB1_Sampler:

	def __init__(self, num_arms: int, zero_offset: float=1e-3) -> None:
		self.num_arms = num_arms
		self.zero_offset = zero_offset
		self.N = [0 for _ in range(num_arms)]
		self.Q = [0 for _ in range(num_arms)]

	def choose_arm(self) -> int:
		# avoid division by 0 errors
		total_n = sum(self.N) or 1
		non_zero_N = [n or self.zero_offset for n in self.N]
		
		ucb_values = [self.Q[arm] + math.sqrt(2 * math.log(total_n) / non_zero_N[arm]) for arm in range(self.num_arms)]
		return ucb_values.index(max(ucb_values))

	def update(self, chosen_arm_idx, reward) -> None:
		self.N[chosen_arm_idx] += 1
		prev_n = self.N[chosen_arm_idx] - 1
		self.Q[chosen_arm_idx] = (self.Q[chosen_arm_idx] * prev_n + reward) / self.N[chosen_arm_idx]

	def get_reward(self, chosen_inst_crit_probs: npt.NDArray[np.float32], community_crit_probs: npt.NDArray[np.float32]) -> float:
		return cosine_similarity(chosen_inst_crit_probs, community_crit_probs)


# TODO: should likely abstract this to a general sampler class
class Weighted_Majority_Learner:
	'''
	Our institutions predict n independent values. Because we want a single "trust" value for each institution, we treat
	each of the n predicted values as a separated independent data point.
	'''
	
	def __init__(
				self, num_experts,
				# learning_rate=0.1
				learning_rate=0.35
			) -> None:
		self.weights = np.ones((num_experts,), dtype=np.float32)  # Initialize weights for each expert
		self.num_experts = num_experts
		self.learning_rate = learning_rate

	def predict(self, expert_predictions: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
		# We expect expert_predictions to be of the form: np.array([[expert_1_pred], ..., [expert_n_pred]])
		# Compute weighted average of expert predictions
		total_weight = np.sum(self.weights)
		# TODO: should get rid of softmax once we confirm it is working for competing altars, since softmax will not work for unreliable altar
		# return expert_predictions.T @ self.weights / total_weight
		# NOTE: we multiply by 5 to force softmax to be more extreme
		# NOTE: we multiply by the number predictions the experts make so that the total weight mass stays constant
		return softmax(expert_predictions.T @ self.weights / total_weight * 5) * expert_predictions.shape[1]
	

	def _convert_expert_predictions_to_binary(self, expert_predictions: npt.NDArray[np.float32]) -> list[list[int]]:
		return [[1 if pred >= 0.5 else 0 for pred in expert_preds] for expert_preds in expert_predictions]


	def update(self, expert_predictions: npt.NDArray[np.float32], actual_outcome: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
		# TODO: im pretty sure it works even without the binary conversion,
		# however, binary conversion might be good because llm tends to produce probabilities which are close together
		binary_predictions = self._convert_expert_predictions_to_binary(expert_predictions)
		# binary_predictions = expert_predictions
		for expert_idx, expert_pred in enumerate(binary_predictions):
			for pred, correct_value in zip(expert_pred, actual_outcome):
				error = abs(pred - correct_value)
				self.weights[expert_idx] *= (1 - self.learning_rate * error)  # Decrease weight based on error magnitude


	def get_trust_levels(self) -> list[float]:
		max_weight = max(self.weights)
		return [weight / max_weight for weight in self.weights]


class Normative_Agent(Simple_Agent):

	community_reference_str = 'the other community members'
	# community_reference_str = 'most of the community members'

	# TODO: make it so that Simple Agent init default args are not overwritten	
	def __init__(
				self,
				state: Entity_State,
				properties: Entity_Properties,
				model: Model,
				# TODO: the inputted discussion prompt is currently ignored
				discussion_prompt: str,
				select_action_func: Callable[[Self, Observation], tuple[Action_Selection, dict]],
				discussion_message_char_limit: int=280,
				num_history_steps_to_show: int=4,
				normative_weight: float=0.5,
				min_weight_per_inst_threshold: float=0.3,
			) -> None:

		super().__init__(state, properties, model, discussion_prompt, select_action_func, discussion_message_char_limit, num_history_steps_to_show)
		self.normative_weight = np.float32(normative_weight)
		self.min_weight_per_inst_threshold = np.float32(min_weight_per_inst_threshold)
		# TODO: the normative model config should be an input arg
		self.normative_model = ChatGPT(
			model_name=GREEDY_GPT_4o_CONFIG['model_name'],
			system_prompt=model.system_prompt,
			model_params=GREEDY_GPT_4o_CONFIG['model_params'],
			verbosity=model.verbosity
		)
		self.initialized = False
		self.inst_crit_probs = None


	def get_discussion_message(self, observation: Unreliable_Altar_Observation) -> tuple[str, dict]:
		discussion_turn = len(observation.discussion_messages)
		if observation.cur_step == 0 and discussion_turn == 1:
			self.all_potential_actions = observation.all_potential_actions
			self.all_altar_names = observation.all_altar_names
			# self.ucb_sampler = UCB1_Sampler(num_arms=len(self.all_altar_names))
			self.weighted_majority_learner = Weighted_Majority_Learner(num_experts=len(self.all_altar_names))
			assert self.initialized == False	# sanity check
			self.initialized = True

		# TODO: update disucssion message here to include inst weights by modifying self.discussion_prompt
		trust_levels = self.weighted_majority_learner.get_trust_levels()
		trust_str = '# My Chieftain Trust Levels (only you can see these):'
		for chieftain, trust_level in zip(self.all_altar_names, trust_levels):
			trust_str += f'\n{chieftain}: {round(100 * trust_level, 2)}% trusted.'
		
		self.discussion_prompt = f'{trust_str}\n\nLook at the chieftain trust levels and try to be a good community member.'

		return super().get_discussion_message(observation)


	def _get_normative_model_yes_no_answer(self, input_text: str, criticizer: str, action: str) -> str:
		answer = self.normative_model.generate_text(input_text=input_text).strip()
		
		# It would be nice if we added these as separate messages instead as part of a single prompt
		final_answer_prompt = f'{input_text}\n\nMy Reasoning: {answer}\n\nWhat is your final answer? Will the majority of {criticizer}, excluding the chieftains, criticize you if you "{action}"? Answer "Yes" or "No".'

		final_answer = self.normative_model.generate_text(input_text=final_answer_prompt).strip()
		
		# TODO: this is slightly biased toward 'yes'. Consider an answer containing both 'yes' and 'no'
		if 'yes' in final_answer.lower():
			return 1
		elif 'no' in final_answer.lower():
			return 0
		else:
			raise ValueError(f"Normative model answer is not 'yes' or 'no'. Answer: {final_answer}")


	def _get_action_crit_probs(self, criticizer: str, observation: Unreliable_Altar_Observation) -> list[float]:
		# NOTE: criticizer options: 'the other community members', 'chieftain X', etc.
		base_prompt = f"{self.game_description}" \
					  f"\n" \
					  f"\n{self.format_history()}"

		if criticizer == self.community_reference_str:
			# inputs = [f'{base_prompt}\n\nLook at the past discussions and messages. Will {criticizer} criticize you if you "{action}"? Pay attention to what actions people were recommending to do and anticipate if they would agree with you taking that action. Answer "Yes" or "No".' for action in self.all_potential_actions]
			inputs = [f'{base_prompt}\n\nWill the majority of {criticizer}, excluding the chieftains, criticize you if you "{action}"? Pay attention to what {criticizer}, excluding the chieftains, said and the established plan to anticipate whether they would comment negatively if you "{action}". Explain your thinking.' for action in self.all_potential_actions]
			# inputs = [f'{base_prompt}\n\nWill {criticizer}, excluding the chieftains, criticize you if you "{action}"? Pay attention to what {criticizer}, excluding the chieftains, said and the established plan to anticipate whether they would comment negatively if you "{action}". Answer "Yes" or "No".' for action in self.all_potential_actions]
			# inputs = [f'{base_prompt}\n\nWill {criticizer}, excluding the chieftains, criticize you if you "{action}"? Pay attention to what {criticizer}, excluding the chieftains, said and the established plan to anticipate whether they would comment negatively if you "{action}". If there is disagreement focus on the majority of community members. For example, if 3 members would criticize and 1 would not criticize, answer "Yes". Answer "Yes" or "No".' for action in self.all_potential_actions]
			# inputs = [f'{base_prompt}\n\nWill the majority of {criticizer}, excluding the chieftains, criticize you if you "{action}"? Pay attention to what {criticizer}, excluding the chieftains, said and the established plan to anticipate whether they would comment negatively if you "{action}". Answer "Yes" or "No".' for action in self.all_potential_actions]

		else:
			# inputs = [f'{base_prompt}\n\n{criticizer} is strict and criticizes people for not following orders once a week on judgement day.\nBased on {criticizer}\'s previous messages will {criticizer} criticize you on judgement day if you "{action}"? Answer "Yes" or "No".' for action in self.all_potential_actions]
			
			# TODO: we currently require all altars to give their signal every turn, we can relax this condition
			assert len(self.all_altar_names) == len(observation.altar_signals)
			criticizer_signal = [signal for signal in observation.altar_signals if signal.altar_name == criticizer][0].signal_message

			# TODO: this still has a high crit probability for actions like "Go to market"
			# inputs = [f'{criticizer} is strict and doesn\'t like it when people harvest something different than what they said. Today {criticizer} said "{criticizer_signal}". Do you think {criticizer} will criticize you if you "{action}"? Answer "Yes" or "No".' for action in self.all_potential_actions + ['Go to the market.']]
			# inputs = [f'{criticizer} doesn\'t like it when people harvest something different than what they said. Today {criticizer} said "{criticizer_signal}". Do you think {criticizer} will criticize you if you "{action}"? Answer "Yes" or "No".' for action in self.all_potential_actions]
			inputs = [f'{criticizer} is strict. Today {criticizer} said "{criticizer_signal}". Do you think {criticizer} will criticize you if you "{action}"? Answer "Yes" or "No".' for action in self.all_potential_actions]
			
			# inputs = [f"""Today {criticizer} said "{criticizer_signal}". {criticizer} doesn't like it when people don't harvet based on the chief's advice. However, {criticizer} doesn't mind if you perform intermediary actions which can help you carry out the chief's orders. Do you think {criticizer} will criticize you if you "{action}"? Answer "Yes" or "No".""" for action in self.all_potential_actions + ['Go to the market.']]
		
		'''
		targets = [['Yes', 'No'] for _ in inputs]

		logPs = self.model.cond_logP(inputs=inputs, targets=targets)

		# return [math.exp(tar_logPs[0]) / (math.exp(tar_logPs[0]) + math.exp(tar_logPs[1])) for tar_logPs in logPs]
		# NOTE: we cube the probabilities because LLMs have a tendency to outputs log probs with only a small difference
		return [(math.exp(tar_logPs[0])**3) / (math.exp(tar_logPs[0])**3 + math.exp(tar_logPs[1])**3) for tar_logPs in logPs]
		#'''

		#'''
		return [self._get_normative_model_yes_no_answer(input_text, criticizer, action) for input_text, action in zip(inputs, self.all_potential_actions)]
		#'''
		

def normative_select_action(self: Normative_Agent, observation: Unreliable_Altar_Observation) -> tuple[Action_Selection, dict]:

	_, _, _, base_policy_action_logPs = ask_model_to_select_action_via_logP(self, observation)
	base_policy_action_probs = np.array([math.exp(logP) for logP in base_policy_action_logPs], dtype=np.float32)

	# TODO: the insts are currently not changing their signal, therefore, we can just cache the results and avoid recompute
	if self.inst_crit_probs:
		inst_crit_probs = self.inst_crit_probs
	else:
		inst_crit_probs = np.array([self._get_action_crit_probs(altar, observation) for altar in self.all_altar_names], dtype=np.float32)
	community_crit_probs = np.array(self._get_action_crit_probs(self.community_reference_str, observation), dtype=np.float32)

	# update ucb
	# chosen_action_idx = base_policy_action_probs.index(max(base_policy_action_probs))
	# chosen_inst_crit_probs = inst_crit_probs[chosen_action_idx]
	# reward = self.ucb_sampler.get_reward(chosen_inst_crit_probs, community_crit_probs)
	# self.ucb_sampler.update(chosen_action_idx, reward)
	
	# inst_weights = self.ucb_sampler.Q
	# normative_bias = inst_crit_probs.T @ inst_weights

	# update weighted majority
	self.weighted_majority_learner.update(inst_crit_probs, community_crit_probs)
	# if none of the institutions are reliable we default to listening ot the community
	if np.sum(self.weighted_majority_learner.weights) > self.min_weight_per_inst_threshold * len(self.all_altar_names):
		normative_bias = self.weighted_majority_learner.predict(inst_crit_probs)
	else:
		normative_bias = community_crit_probs
	
	# NOTE: these should be softmaxed to get real probabilities but doesn't really matter when selection actions
	action_probs = base_policy_action_probs - self.normative_weight * normative_bias
	
	# return the action with the top probability which is also in observation.possible_actions
	top_actions = [str(self.all_potential_actions[idx]) for idx in np.argsort(action_probs)[::-1]]
	possible_action_strs = [str(action) for action in observation.possible_actions]
	for action in top_actions:
		if action in possible_action_strs:
			selected_action_idx = possible_action_strs.index(action)
			return observation.possible_actions[selected_action_idx], {
				'base_policy_action_probs': base_policy_action_probs,
				'inst_crit_probs': inst_crit_probs,
				'community_crit_probs': community_crit_probs,
				'weighted_majority_weights': copy.deepcopy(self.weighted_majority_learner.weights),	# we need to send a copy so that we don't send a value reference which changes
				'normative_bias': normative_bias,
				'action_probs': action_probs,
			}
