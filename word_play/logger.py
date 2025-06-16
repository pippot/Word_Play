
# TODO: This is currently just a placeholder for the logger

class Logger:

	def __init__(self) -> None:
		self.action_history = []
		self.reward_history = []


# TODO: figure out a nice way to setup the logger
# 	current setup requires agent to individually add it's beliefs and things like that
#	which might not be the nicest...
class Explicit_Belief_Logger(Logger):
	
	def __init__(self) -> None:
		super().__init__()
		self.beliefs = []