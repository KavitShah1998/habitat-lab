# [NAOKI]
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction
import habitat_sim
import habitat

import torch
import numpy as np

HabitatSimActions.extend_action_space("CONT_MOVE")

import attr
@attr.s(auto_attribs=True, slots=True)
class ContCtrlActuationSpec:
    pass

@habitat_sim.registry.register_move_fn(body_action=True)
class NothingAction(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: ContCtrlActuationSpec,
    ):
        pass

@habitat.registry.register_action_space_configuration
class ContCtrlSpace(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.CONT_MOVE: habitat_sim.ActionSpec("nothing_action"),
        }

@habitat.registry.register_task_action
class ContMove(SimulatorTaskAction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Icky, but apparently necessary...
        self._robo_id_1 = 0

    def _get_uuid(self, *args, **kwargs) -> str:
        return "cont_move"

    def reset(self, task, *args, **kwargs):
        task.is_stop_called = False
        if not self._sim._sim.get_existing_object_ids():
            self._sim._sim.config.sim_cfg.allow_sliding = True
            obj_templates_mgr = self._sim._sim.get_object_template_manager()
            # locobot_template_id = obj_templates_mgr.load_object_configs('/nethome/nyokoyama3/habitat-sim/data/objects/locobot_merged')[0]
            locobot_template_id = obj_templates_mgr.load_object_configs('/coc/pskynet3/nyokoyama3/aug26v2/habitat-sim/data/test_assets/objects/sphere')[0]
            self._robo_id_1 = self._sim._sim.add_object(locobot_template_id, self._sim._sim.agents[0].scene_node)
            self._sim._sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, 0)

    def step(
    	self,
    	task,
    	linear_velocity,
    	angular_velocity,
    	time_step,
    	allow_sliding=True
    ):
        self._sim._sim.config.sim_cfg.allow_sliding = allow_sliding

        if (
        	abs(linear_velocity) < 0.1*0.25
        	and abs(angular_velocity) < 0.1*np.pi/18.0
        ):
            task.is_stop_called = True
        
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.controlling_ang_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.ang_vel_is_local = True
        vel_control.linear_velocity = np.array([0.0, 0.0, linear_velocity])
        vel_control.angular_velocity = np.array([0.0, angular_velocity, 0])

        # New code start
        previous_rigid_state = self._sim._sim.get_rigid_state(self._robo_id_1)
        # manually integrate the rigid state
        target_rigid_state = vel_control.integrate_transform(
            time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        end_pos = self._sim._sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )
        self._sim._sim.set_translation(end_pos, self._robo_id_1)
        self._sim._sim.set_rotation(target_rigid_state.rotation, self._robo_id_1)

        # Check if a collision occured
        dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
        ).dot()
        dist_moved_after_filter = (end_pos - previous_rigid_state.translation).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        # New code end

        
        ret = self._sim._sim.step_physics(time_step)
        return self._sim.step(HabitatSimActions.CONT_MOVE)