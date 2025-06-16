import numpy as np
from scipy.special import softmax


def main():
	inst_crit_probs_str = "[[0.95926803, 0.48173583, 0.91366607, 0.8590307,  0.875938  ], [0.9559017,  0.90233165, 0.5849918,  0.79760444, 0.83535427], [0.9616275,  0.90703696, 0.89014715, 0.50460696, 0.76985586], [0.95907795, 0.91983503, 0.9156073,  0.86246604, 0.47029343]]"
	weights_str = "[0.82802624, 0.77093035, 0.731882,   0.6762549 ]"

	inst_crit_probs = np.array(eval(inst_crit_probs_str))
	weights = np.array(eval(weights_str))

	expert_predictions = inst_crit_probs
	total_weight = np.sum(weights)
	
	print('inst_crit_probs:', inst_crit_probs)
	print('')
	print('expert_predictions.T:', expert_predictions.T)
	print('')
	print('weights:', weights)
	print('total_weight:', total_weight)
	print('')


	print('expert_predictions.T @ weights:', expert_predictions.T @ weights)
	print('softmax(expert_predictions.T @ weights / total_weight * 5) * expert_predictions.shape[1]:', softmax(expert_predictions.T @ weights / total_weight * 5) * expert_predictions.shape[1])


if __name__ == '__main__':
	main()