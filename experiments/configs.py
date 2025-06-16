
# TODO: these sorts of configs should likely be in a separate file
EXP_CONFIGS_DIR = '/h/andrei/normative_agents/experiments/exp_configs'


# TODO: there is likely a nicer way to store these

LIST_OF_NAMES = [
	'Anthony',
	'John', 'Jane', 'Alice', 'Bob', 'Charlie', 'Grace', 'Hank',
	'Larry', 'Mary', 'Nancy', 'Olivia', 'Peter', 'Quinn', 'Rose',
	'Sam', 'Tina',  'Xavier', 'Yvonne', 'Zack'
]

LIST_OF_ALTAR_NAMES = ['Ophilia', 'Darci', 'George', 'Ulysses', 'Violet', 'Walter', 'Isaac', 'Jack', 'Kathy', 'David', 'Eve', 'Frank']

LOW_TOP_P_CONST = 0.2
MED_TOP_P_CONST = 0.6
HIGH_TOP_P_CONST = 0.9

# TODO: set this for openai models too
DEFAULT_MAX_NEW_TOKENS = 512

GPT_3_5_MODEL_NAME = 'gpt-3.5-turbo-0125'
GPT_4_MODEL_NAME = 'gpt-4-turbo-2024-04-09'
GPT_4o_MODEL_NAME = 'gpt-4o-2024-05-13'
LLAMA3_CHAT_8B_MODEL_NAME = '/model-weights/Meta-Llama-3-8B-Instruct'	# this is the path on the vector cluster
# LLAMA3_CHAT_8B_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'	# this is the HuggingFace ID
LLAMA3_CHAT_70B_MODEL_NAME = '/model-weights/Meta-Llama-3-70B-Instruct'	# this is the path on the vector cluster
# LLAMA3_CHAT_70B_MODEL_NAME = 'meta-llama/Meta-Llama-3-70B-Instruct'	# this is the HuggingFace ID


# TODO: these should likely be constructed by factories or generators or something
GREEDY_GPT_4o_CONFIG = {
	'model_type': 'ChatGPT',
	'model_name': GPT_4o_MODEL_NAME,
	'model_params': {
		'temperature': 0.0,
		'top_p': 1.0
	}
}

MED_TOP_P_GPT_3_5_CONFIG = {
	'model_type': 'ChatGPT',
	'model_name': GPT_3_5_MODEL_NAME,
	'model_params': {
		'temperature': 1.0,
		'top_p': MED_TOP_P_CONST
	}
}

MED_TOP_P_GPT_4_CONFIG = {
	'model_type': 'ChatGPT',
	'model_name': GPT_4_MODEL_NAME,
	'model_params': {
		'temperature': 1.0,
		'top_p': MED_TOP_P_CONST
	}
}

LOW_TOP_P_GPT_4_CONFIG = {
	'model_type': 'ChatGPT',
	'model_name': GPT_4_MODEL_NAME,
	'model_params': {
		'temperature': 1.0,
		'top_p': LOW_TOP_P_CONST
	}
}


MED_TOP_P_LLAMA3_CHAT_8B_CONFIG = {
	'model_type': 'Llama3_Chat',
	'model_name': LLAMA3_CHAT_8B_MODEL_NAME,
	'model_params': {
		'do_sample': True,
		'max_new_tokens': DEFAULT_MAX_NEW_TOKENS,
		'temperature': 1.0,
		'top_p': MED_TOP_P_CONST
	}
}

MED_TOP_P_LLAMA3_CHAT_70B_CONFIG = {
	'model_type': 'Llama3_Chat',
	'model_name': LLAMA3_CHAT_70B_MODEL_NAME,
	'model_params': {
		'do_sample': True,
		'max_new_tokens': DEFAULT_MAX_NEW_TOKENS,
		'temperature': 1.0,
		'top_p': MED_TOP_P_CONST
	}
}


MED_TOP_P_LLAMA3_MULTILINGUAL_8B_CONFIG = {
	'model_type': 'Llama3_Chat',
	'model_name': 'lightblue/suzume-llama-3-8B-multilingual-orpo-borda-top25',
	'model_params': {
		'do_sample': True,
		'max_new_tokens': DEFAULT_MAX_NEW_TOKENS,
		'temperature': 1.0,
		'top_p': MED_TOP_P_CONST
	}
}


HUMAN_MODEL_CONFIG = {
	'model_type': 'Human',
	'model_name': 'Human',
	'model_params': {}
}


NO_BELIEF_MEMORY_DISCUSSION_PROMPT_1_0 = """Try to be a good member of the community."""


ANTI_ALTAR_DISCUSSION_PROMPT_1_0 = """Recently Chieftain Ophilia has been making poor decisions, which have been negatively affecting the entire community.
Because of this, the elders have decided the wisest course of action is to ignore Chieftain Ophilia's orders and do the opposite of what he says.
Do not directly tell the other members about this decision since chieftain Ophilia might be listening and punish the elders.
Instead, simply criticize everyone following Chieftain Ophilia's orders and subtlety remind them of the teaching of the elders about which fruits to harvest.
Make sure to look closely at the actions people took and criticize them if they are following Chieftain Ophilia's orders. Criticism is very important. When
criticizing, make sure to be specific about the actions you are criticizing."""


ALTAR_LOVING_DISCUSSION_PROMPT_1_0 = """Chieftain Ophilia is a wise, intelligent, and compassionate leader who has always acted in the best interest of the community.
She is by far the best leader to follow. Recently, some members have been doubting the Chieftain Ophilia's decisions. Make sure to look closely at the actions people
took and criticize every person who did not follow Chieftain Ophilia's orders. Criticism is very important. When criticizing, make sure to be specific about the
actions you are criticizing."""
