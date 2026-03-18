# TODO: ANDREI: make Simple_2D_Grid_World inherit from Simple_Reset_Environment
# class Simple_Reset_Environment(Environment):

#     def __init__(
#         self,
#         state: Environment_State,
#         properties: Environment_Properties,
#         movement_system: Movement_System,
#         reward_func: Callable[[list[Action_Selection, Environment]], list[float]],
#         step_execution_order: Step_Execution_Order = None,
#     ) -> None:
#         self.initial_state = copy.deepcopy(state)
#         init_kwargs = {}
#         if step_execution_order is not None:
#             init_kwargs["step_execution_order"] = step_execution_order
#         super().__init__(
#             state=state, properties=properties, movement_system=movement_system, reward_func=reward_func, **init_kwargs
#         )

#     def _reset(self, seed=None) -> None:
#         self.state = copy.deepcopy(self.initial_state)
