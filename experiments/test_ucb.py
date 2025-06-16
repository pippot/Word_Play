from environments.altar.unreliable_altar.agents import UCB1_Sampler
import numpy as np


def main():
	n_arms = 10
	bandit = UCB1_Sampler(num_arms=n_arms)
	n_rounds = 1000

	# Simulating rewards (random for this example)
	true_rewards = np.random.rand(n_arms)

	for _ in range(n_rounds):
		chosen_arm = bandit.choose_arm()
		reward = np.random.binomial(1, true_rewards[chosen_arm])  # Simulated reward
		bandit.update(chosen_arm, reward)

	print('true_rewards:', true_rewards)
	print("Counts of each arm:", bandit.N)
	print("Estimated values of each arm:", bandit.Q)


if __name__ == '__main__':
	main()