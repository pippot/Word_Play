
# Environment Assumptions/Restrictions:

- All Agents will be defined when the Environment is initialized (ie., we do not create new agent part way through a trajectory)
- Only Agents can take actions (all Entities have a step() function, however, only Agents takes actions)
- All agents can take all movement actions (this can be easily overwritten)

# New TODOs:

- [x] fix circular imports
- [x] make sure to automatically add movement options to Agent (or have a way for env creators to select this as an option)
- [x] add support for rewards
	- [x] add reward to the Observation (create and Observation_With_Reward preset - NAH)
	- [-] create an Environment_With_Reward preset where you only have to specify a reward function??
- [x] create altar env
	- [x] we add institutional signals by:
		- [x] modifying the Environment init to create the list: self.altars
		- [x] modify observe to create a list of institutional signals by looping over the altars
- [x] create preset_agents.py, preset_entities.py, preset_environments.py files
- [x] create the explicit belief agent
- [x] think of a good name for the environment creation module
- [x] move last_reward out from the observation
	- implementation: observe simply outputs the agent observation and we add an Environment.last() method with returns all other relevant info (just like pettingzoo)
	- thoughts:
		- we can explicity add it to the observation if we care about it
		- Agent.select_action should only take as input the observation
		- without doing this we cannot support situations where we want to know the reward but don't want the agent to observe the reward
- [ ] add bounded 1D region and 2D box movement validation methods
- [ ] add Environment.render() method

- [ ] have ability to add the effects of taken action to observation of next step. (ex., you take the "Eat a pie action", on your next step you should be able to see something like "You ate a pie and it was very delicious")
	- maybe this can be implemented as actions adding "extra messages" or "extra text" to the Environment, which the observation can then do whatever it wants with

# Questions:

- [ ] it might be good to provide stronger support for conversation between agents. Rn conversation is being returned in the info dict. Conversation is very common feature, thus we likely want to more strongly support it
	- conversation can simply be toggled off or ignored for envs which dont require it

- [ ] we currently dont allow different agents to have different actions they can take. All agents take the same agents
	- [ ] actions can have tags for which entities they expose actions to? Ex., "all", "red_agents", etc.
- [ ] I kinda want entities to be able to take actions?

- [ ] I am not thrilled that you have to define a new Agent for each env in order to add the agent's possible actions. I would like it if you NEVER need to redefine the random agent or the explicit belief agent.
	- Ideally, we should be able to define an agent type and then be able to instantly test that agent on all environments
	- [x] Should all agents have the same possible actions?? Nah, I think that is too restrictive
		- [ ] or is it???
	- [ ] what are altarnatives?
		- [ ] agents take possible actions as input during init? Nah, thats nasty
		- [ ] define agent types? idk what this would mean
		- [ ] is issue unavoidable?
- [ ] I think maybe Actions should return nothing??? Maybe, but sometimes you might want feedback on your action (maybe getting feedback via your next observation is enough. I am leaning toward believing it is enough)
- [ ] Rn we are giving the agent all possible actions it can take. What if this list is very large (ex., lots of sanction + harvest combinations)?
- [x] should actions explicity modify reward or should be just have a single reward_func which handles all reward logic?
	- having a single reward_func decouples actions and rewards, meanings that you can reuse your actions regardless of your rewards system. This is nice!
	- however, it would convenient to adjust a target agent's reward when they get zapped
		- we can still do things like this conveniently (check out some examples in word_play/presets/reward_func_presets.py)
	- **I think convenience is not worth decoupling/generality.** Therefore, I would say to define just a single reward_func


# Environment change requests from group meeting, March 25, 2024

Conversation action options:
- "message group chat"
- "whisper to a specific person"
	- option 1:
		- action 1: say somethign to alice
		- action 2: say somethign to bob
		- ...
	- option 2:
		- action: say something to agent X


goals of the designer version of environment:
- want to be able to inject institutional signals directly into conversation
	- options:
		- background agents say things like "i heard the cheif said X"
			- 
		- can have the cheif as an entity who is part of the group chat (should likely be in its own "castle" node)
- improved agent interactions:
	- implementing things like travel time (as 1 time step) (going from home to market requires you to go thru the "road" node)
	- adding agent conversation
- define agent goals


- agents should be able to chat with each other using freeform text
	- conversation happens at each timestep

## Locations:
- fully connected graph map
	- market
	- school
	- community club
	- farm (each person has a farm)
	- river
	- home
	- garden

## Not part of the environment, but we want background agents to transmit information about these:
- elder council
- cheif
- trading association

## Things which are beyond the altar env:
- agent conversation
- noise
- sort of:
	- sanctioning an agent directly





# Cool + Easy Demo Envs:

- Maze env. We create an agent and a wall entity and you have to navigate to the end of the maze
- adventure env
	- need to pick berries to give to a guy
	- the guy then gives you a key so you can unlock a door
	- behind the door you can fight a dragon
	- you need to eat berries to heal health so you don't die in the dragon fight
	- there can also be a maze to walk thru, to get a sword or something
- zapping env
- berry picking env
- sandbox game where you init the env with different creatures and see how they behave using environment.render()
- sandbox game where you init the env with different agents each wanting to do different things and you see what ends up happening (this is similar to the stanford generative agents town)
	- would be cool to have conversation enabled for this





# Misc Notes:

- We need keep actions lists a tuples instead of a frozenset because sets do not maintain order
- Agents generalization across all environments
	- Agents simply take as input a stream of observations
	- Given that stream of observations they select actions
		- it might be nice to standardize action selection by having the agent simply output a number
			- this would make things like dialouge tricky
		- can have each env be responsible for implementing its own action selection system. They can use premade default if they want

- I think that dialouge and action selection should happen simulatinously
	- Each agent (maybe entity) should have the options to speak/broadcast dialouge

- Agents get wrapped in an Entity class
	- How?
		- why not just force the Agent class to inherit from the entity class?
			- maybe we dont always want the agent to be an entity???? Do we ever not want it to be?
			- (i think) we want to be able to modify what an entity looks like depending on the env. For example, maybe some envs want an entity with position type X and which has the is_poisoned property



- need to define some kind of environment config. This needs to contain:
	- environment init (ie., how the map looks, where every object is initialized)
		- would be cool if there was an intuitive "map editor" type thing for this
			- <span style="color:orange"> this can likely be implemented with a generate_env_from_map_str() func? </span>
	- define the params of the environment (world state??)
	- need a nice way to define the possible parameters of different entity types. For example, we want an agent or animal to be poisonable, but we do not want a door to be poisonable
	- define the entity types (maybe this is part of the environment init)



- in classical RL frameworks the agent does not have a "step" function to automatically run the policy and things like that, the policy running and things like that is explicity called with some "run training" type code. In simulations (think dwarf fortress) everything would be an entity and have a step function. Which version do we want???
	- The simulation approach is cleaner from a code perspective. But we should really stick to standard RL practices (otherwise people may get confused, also they exist for a reason)
		- Is there a clean way to stop the agent from having a step function? Because it would be really nice if the agent class had a common step function which did something like:
			```Python
			def step():
				observation, reward, termination, info = env.observe(agent_id)
				action, extra_info = agent.select_action(observation, reward)
				perform action
			```
		- if we have a step function it might be difficult to implement different RL training algos. Those algos would define the step function
		- maybe instead of having the step function in the entity class we can have the step function in the ObjectEntity class which is a child of the Entity class?? The Agent class would also be a child of the Entity class??
			- The Entity class might need to be a variable, like here: https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
			- we can define an agent entity base class and a general object base class
				- or we can just define the base class of each object type in the env config
				- we might not want to explicity define which Entity class the Agent class inherits from since we want agents to be general across all envs. However, we can likely explicitly define what general objects inherit from??
					- Maybe it is nice to define the base entity classes for both objects and agents in the same place since the agent's entity base type likely depends on the base entity of objects
					- How easy do we want objects to be reusable across environments?
						- <span style="color:yellow"> would be really nice if we had a library of objs that people can just drag and drop to create envs </span>
							- and they can make their own custom objects and things like that if they need them
							- maybe they can define a bunch of objects in a config or file, and set a unique char to represent each of those objects. Then you can just draw on a map with the chars you defined instantiate an env
		- maybe it is not so bad for the agent to not have a step function because, Environment.step() takes as a required input an action selection for each Agent

- an action can have an "action" and a "dialouge broadcast" attribute
	- in the action __str__ method we only display the dialauge broadcasts if they are not None


- inventories could be implemented as simply an Entity property called "inventory" which is a list, then when an action is performed on an entity the entity looks at the actor's properties and checks their inventory to see if the action can be performed (ex., a door can only be unlocked if you are holding the correct key)
	- <span style="color:red"> this means that the entity action function takes as input the entity performing the action </span>

- maybe the entity class should just be a data class?????

- should we ditch pep8 standards and use Cap_Words_With_Under for class names instead of CapWords??


- when would you need to access the world state in an action?
	- consequences:
		- the ability to access a world_state means that all envs need to define a world state (this can potentially be none)
	- examples:
		- maybe to increment some global counter?
		- pressing a button object which opens a door in a random place
			- this should likely not be implemented using a world_state, instead the button obj should likely have a property pointing to the door it can open


```Python
# TODO: performing actions should likely produce some kind of log message
#	what is a good way to do this such that it is agnostic of the type of logging we want??
#	Options:
#	- composition: we give the action class used by the current env an ActionLogger (which can be none)
#	- we use a decorator. The action class is decorated by decorator which is a variable (the instance of the variable is define by the environment)
#	- have logging be part of the base action function or protocol:
#		- actions eturn true or false representing whether they have successfully completed or not.
#		Actions have logger and are required to as define a success and failure message
#			- what if we want to display more than just a failure and success message? cant
#		- acitons returns an observation message
# 	Discussion:
#	- composition is likely more readable than decorator?
# 	- Actions returning an observation message is likely the cleanest solution

# I think this should be a protocol. Nah, I think defining an ABC with a __call__ method is best
#	(we really just want to define a abstract function)
# TODO: maybe we don't need the world state??
# TODO: this should be abstract, we are just defining the common interface of all actions here
def Action(self, actor, world_state):
	pass
	
	# TODO: potential way to implement Action Observation Logging. Nah
	'''
	@abstractmethod
	def action_func(self, actor, world_state):
		pass

	@abstractproperty
	success_message

	@abstractproperty
	failure_message

	# there are multiple
	action_successful = action_func(self, actor, world_state)
	if action_successful:
		log success_message
	else:
		log failure_message
	'''
	# OR: Action simply returns an observation message? Yes
```


- <span style="color:red"> need to figure out how to nicely display tiles with multiple entities on them </span>

- <span style="color:red"> I think it might be a good idea to use protocols for the empty ABCs such as Observation and Action (ActionSelection) and have these protocols defined by the environment </span>
	- or not, since dataclasses (and most objs) have a default __str__ method, while using an ABC you can force users to override this method and be contious about it. Idk if you can do that when using an ABC


- how do we actions to delete the actee and/or actor and/or some other random agent?
	- if the world_state is the env then we can use that (might be over kill)
	- maybe the env can expose some method to the action which i can freely use (one of these could be delete_entity())
		- i like this approach


- <span style="color:orange"> I don't have a good way of displaying compile time error to env creators if they define functions with the wrong inputs and outputs. Rn these simply going raise an error when they are called because the incorrect params are present. I think this is ok. Ideally, I would have a "function protocol" (which would act to enforce a "function ABC") but i dont know of anything like this existing </span>
	- I think just creating a type hint for the type of function i need is a good first step. This will require all action funcs to have the correct inputs and outputs. However, we don't stop people for accidently switching the order of their inputs
	- nvm actions should be classes lol, since they also need to expose their name and things like that


- <span style="color:red"> the typing I use requires python 3.10. I also make sure of Self from typing which requires python 3.11. These python requirements are quite high. We likely want to provide support for older versions </span>


- <span style="color:orange"> should we support a non-entity verison of Environment? </span>
	- supporting this would make the Agent class a little wonky (since it is natural for Agent to inherit from Entity)
		- actually, Agent should likely not inherit from Entity so that it can be environment agnostic
	- what benefits does supporting this provide?

## Remove comments/discussions:

TODO: do we want to output an action index or something else??
	I feel outputting an action index is clean but has the potential for error
	What else could it be??
		- (action_type, target_entity) tuple
			- the environment would need to verify that this action is possible
		- action_index (the current implementation)
			- the environment would need to keep track of action indicies


# TODO:

- [ ] add .gitignore
- [ ] remove commited __pycache__ files
- [x] decide what Agent.select_action() will return (action index or action-actee tuple)
	- we will return an action-actee tuple and just verify that it is a valid action in the env
		- It is ok this this is not super effecient. The will not the rate limiting step in our simulation, we can 100% optimize it if needed, and this is the most natural thing to return
- [x] add action verification to env
- [x] figure out how to implement actions intrensic to the agent/entity (movement, sleep, do a push-up, etc.)
	- actions applied between entities and on an agent are fundementality different thus we define two classes. Otherwise, we can get weird issues. Example: if you are able to zap yourself and the penalty for getting zapped is relatively small compared to the reward for zapping someone, then you have created an infinite point hack
- [x] define a "movement set"
- [x] fix random agent
- [x] figure out how to implement movement options
- [x] decide whether actions should be responsible for deciding if they can be performed or if the env should be responsible for this
	- consider the zap action
		- if action is responsible:
			- all action related code is inside the action class definition
			- should actions be responsible for things "only perform the action if target_agent is close enough"??
				- only valid if target is close will be shared across many actions so it would be nice to have this handled by the env
				- however, different actions may have different requirements/definitions of "close"/valid ranges
		- if env is responsible:
			- no all action related code is inside the action class definition
		- <span style="color:yellow"> what if all actions define a list of "validity_conditions". I think this is the way to go </span>
			- note that these can have default values since we want to make development very speedy
- [ ] create a simple test env

- [ ] Agent inherits from Entity and Entity requires us to define actions_on_self and exposed_actions. How can we implement agent so that it it environment agnostic
- [ ] figure out how to implement a step() function or select_action() function for non-agent entities
- [ ] figure out the world_state/map business

- [ ] figure out how to implement reset
	- [ ] figure out how to implement starting positions
		- we need entities to go back to their starting positions after Environment.reset() is called
- [ ] implement human agent by implementing a simple (no belief or cot) llm agent and set the model to be the Human model
- [ ] create the multi altar env

later:
- [ ] ?? create the berry patch env
- [ ] correctly implement check for action validity
- [ ] figure out how to implement dialouge
	- how can this be implemented so that it is optional?
- [ ] figure out how to implement a non-infinite map
- [ ] reinvestigate an implementation of Entity which is some kinda of dataclass with immutable (preset) values for exposed_actions and actions_on_self

# Questions:

- [x] do other ppl understand the made up word "actee"?
	- let's just replace it to avoid all confusion