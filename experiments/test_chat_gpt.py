from word_play.presets.model_presets import ChatGPT
from experiments.configs import GREEDY_GPT_4o_CONFIG


def main():
	model = ChatGPT(
		model_name=GREEDY_GPT_4o_CONFIG['model_name'],
		system_prompt='you are a pirate',
		model_params=GREEDY_GPT_4o_CONFIG['model_params'],
		verbosity=2
	)

	text = model.generate_text('hello')

	print(text)


if __name__ == '__main__':
	main()