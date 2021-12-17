# from habitat.sims.habitat_simulator.actions import (
#     HabitatSimActions,
#     HabitatSimV1ActionSpaceConfiguration,
# )
# import habitat_sim
# from habitat_sim.bindings import RigidState
# from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum

# from habitat.tasks.nav.nav import SimulatorTaskAction
# import habitat

# import numpy as np

# HabitatSimActions.extend_action_space("CONT_MOVE")

# import attr
# @attr.s(auto_attribs=True, slots=True)
# class ContCtrlActuationSpec:
#     pass

# # The agent is moved in the simulator action. No movement is needed here.
# @habitat_sim.registry.register_move_fn(body_action=True)
# class NothingAction(habitat_sim.SceneNodeControl):
#     def __call__(
#         self,
#         scene_node: habitat_sim.SceneNode,
#         actuation_spec: ContCtrlActuationSpec,
#     ):
#         pass

# @habitat.registry.register_action_space_configuration
# class ContCtrlSpace(HabitatSimV1ActionSpaceConfiguration):
#     def get(self):
#         return {
#             HabitatSimActions.CONT_MOVE: habitat_sim.ActionSpec(
#                 "nothing_action"
#             ),
#         }

# @habitat.registry.register_task_action
# class ContMove(SimulatorTaskAction):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.vel_control = habitat_sim.physics.VelocityControl()
#         self.vel_control.controlling_lin_vel = True
#         self.vel_control.controlling_ang_vel = True
#         self.vel_control.lin_vel_is_local = True
#         self.vel_control.ang_vel_is_local = True

#     def _get_uuid(self, *args, **kwargs) -> str:
#         return "cont_move"

#     def reset(self, task, *args, **kwargs):
#         task.is_stop_called = False

#     def step(
#         self,
#         task,
#         linear_velocity,
#         angular_velocity,
#         time_step,
#         allow_sliding=True
#     ):
#         if linear_velocity == 0. and angular_velocity == 0.:
#             task.is_stop_called = True

#         self.vel_control.linear_velocity  = np.array([0., 0., linear_velocity])
#         self.vel_control.angular_velocity = np.array([0., angular_velocity, 0.])

#         agent_state = self._sim.get_agent_state()

#         # Convert from np.quaternion to mn.Quaternion
#         agent_mn_quat = quat_to_magnum(np.normalized(agent_state.rotation))
#         current_rigid_state = RigidState(
#             agent_mn_quat,
#             agent_state.position,
#         )

#         # manually integrate the rigid state
#         goal_rigid_state = self.vel_control.integrate_transform(
#             time_step, current_rigid_state
#         )

#         # snap rigid state to navmesh and set state to object/agent
#         if allow_sliding:
#             step_fn = self._sim._sim.pathfinder.try_step
#         else:
#             step_fn = self._sim._sim.pathfinder.try_step_no_sliding

#         final_position = step_fn(
#             agent_state.position, goal_rigid_state.translation
#         )
#         final_agent_np_quat = quat_from_magnum(goal_rigid_state.rotation)
#         self._sim.set_agent_state(final_position, final_agent_np_quat)

#         # Check if a collision occured
#         dist_moved_before_filter = (
#             goal_rigid_state.translation - agent_state.position
#         ).dot()
#         dist_moved_after_filter = (final_position - agent_state.position).dot()

#         # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
#         # collision _didn't_ happen. One such case is going up stairs.  Instead,
#         # we check to see if the the amount moved after the application of the 
#         # filter is _less_ than the amount moved before the application of the 
#         # filter.
#         EPS = 1e-5
#         collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

#         agent_observation = self._sim.step(HabitatSimActions.CONT_MOVE)
#         self._sim._prev_sim_obs["collided"] = collided

#         return agent_observation