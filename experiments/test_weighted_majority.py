from environments.altar.unreliable_altar.agents import Weighted_Majority_Learner
import numpy as np


def main():
	# 3 columns, 4 rows
	test_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
	print('test_arr:', test_arr)
	print('test_arr.shape:', test_arr.shape)

	expert_predictions = [
							np.array([[1, 0, 0, 0],		# Expert 1's prediction
				 					  [0, 1, 0, 0], 	# Expert 2's prediction
									  [0, 0, 1, 0], 	# Expert 3's prediction
									  [0, 0, 0, 1], 	# Expert 4's prediction
									  [1, 1, 1, 1]]), 	# Expert 5's prediction
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
							np.array([[1, 0, 0, 0],
				 					  [0, 1, 0, 0],
									  [0, 0, 1, 0],
									  [0, 0, 0, 1],
									  [1, 1, 1, 1]]),
						]

	actual_labels = [
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
						np.array([1, 0, 1, 0]),
					]

	# desired weights = [1, 0, 1, 0, 0]

	learner = Weighted_Majority_Learner(num_experts=5)

	for t in range(len(actual_labels)):
		prediction = learner.predict(expert_predictions[t])
		print(f"Round {t + 1}:")
		print(f"Expert predictions: {expert_predictions[t]}")
		print(f"Algorithm prediction: {prediction}")
		print(f"Actual label: {actual_labels[t]}")
		learner.update(expert_predictions[t], actual_labels[t])
		print(f"Updated weights: {learner.weights}\n")


if __name__ == '__main__':
	main()