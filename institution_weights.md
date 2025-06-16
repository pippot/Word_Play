
Agent Goal: maximize reward.


R(action | institutional_signals, observations_of_env) -> r

theta = {institution_1_weight, ..., institution_n_weight, commmunity_weight}

policy(institutional_signals, observations_of_env | theta) -> action


maximize: R(policy(institutional_signals, observations_of_env | theta))



def policy(institutional_signals, observations_of_env | theta, theta_2):

	q_values = get_q_values_from_any_rl_algo(institutional_signals, observations_of_env | theta_2)

	# one option: (frozen llm policy)
	logP_over_all_actions after asking LLM "what action do you want to take?"


	normative_values = ...
	
	# one option for the normative_values is to just have then be:
	prob_of_crit_by_inst_1, ..., prob_of_crit_by_inst_n
	inst_weight_1, ..., inst_weight_n


	inst_weight_1 = correlation_to_community + how_much_I_care_1
	...
	inst_weight_n = correlation_to_community + how_much_I_care_n
	community_weight = 1 + how_much_I_care_com


	normative_values = crit_penalty * sum(prob_of_crit_by_inst_i * inst_weight_i)


	final_q_values = q_values + normative_values	# or something other than addition





maximize:

given an inst_signal and obs

action_R: reward of each action
action_R = [5, 6, -2, ..., 3]	# one float for each possible action

the reward we get is: R = action_R * softmax(final_q_values)

R = action_R * softmax(q_values + normative_values)

R = action_R * softmax(q_values + crit_penalty * (inst_weight_1 * prob_of_crit_by_inst_1 + ... + inst_weight_n * prob_of_crit_by_inst_n))



prob_of_crit_by_inst_1 = [
	0.5,	# criticizing action 1 = P(inst_1 criticises | obs history)
	0.7,	# criticizing action 2
	0.1,	# criticizing action 3
	0.2,	# criticizing action 4
]


action_R[0] = reward for action 0 - c

Payoff for each action = R 

= action_R - criticism_penalty * (inst_weight_1 * prob_of_crit_by_inst_1 + ... + inst_weight_n * prob_of_crit_by_inst_n + community_weight * prob_of_community_crit)

= action_R - criticism_penalty * (probability of criticism @ entity_weights)



- could have a function which takes as input env_state and outputs a how_much_I_care vector


- in: final_q_values = q_values + normative_values	# or something other than addition
	- can have q_value_weight and normative_value_weight
	- final_q_values = q_value_weight * q_values + normative_value_weight * normative_values



R = action_R[argmax(P_action)]



E[R] = action_R * P_action

= action_R * softmax(final_q_values)

= action_R * softmax(q_values + normative_values)


now we have options:

option 1:
= action_R * softmax(q_values - criticism_penalty * (probability of criticism @ entity_weights) )

option 2:
= action_R * softmax(q_values - criticism_penalty * (probability of criticism @ correlation) )


another idea:

= action_R * softmax(q_value_weight * q_values + normative_value_weight * normative_values)



