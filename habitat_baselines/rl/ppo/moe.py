from habitat_baselines.rl.ppo.policy import (
    Policy,
    PointNavBaselineNet,
)
from habitat_baselines.rl.ppo.sequential import (
    ckpt_to_policy,
    get_blank_params,
)
from gym.spaces import Box, Dict
import torch

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs, ObservationBatchingCache

"""

Focus first on NavGaze MoE....
"""

ARM_ACTIONS = 4  # 4 controllable joints
BASE_ACTIONS = 2  # linear and angular vel
EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"
EXPERT_MASKS_UUID = "expert_masks"


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsRes(Policy):
    def __init__(
        self,
        observation_space: Dict,
        action_space,
        nav_checkpoint_path,
        gaze_checkpoint_path,
        goal_hidden_size,
        fuse_states,
        num_environments,
        hidden_size: int = 512,
        train_critic_only=False,
        *args,
        **kwargs,
    ):
        # Add expert actions to the observation space and fuse_states
        observation_space.spaces[EXPERT_NAV_UUID] = Box(
            -1.0, 1.0, (BASE_ACTIONS,)
        )
        observation_space.spaces[EXPERT_GAZE_UUID] = Box(
            -1.0, 1.0, (ARM_ACTIONS,)
        )
        fuse_states.extend([EXPERT_NAV_UUID, EXPERT_GAZE_UUID])

        # Instantiate MoE's own policy
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_hidden_size=goal_hidden_size,
                fuse_states=fuse_states,
                force_blind=False,
            ),
            action_space,
        )

        # For RolloutStorage in ppo_trainer.py
        self.policy_action_space = action_space

        # Get model params before experts get loaded in
        self.model_params = [p for p in self.parameters() if p.requires_grad]

        # Freeze non-critic weights if training only critic
        if train_critic_only:
            for name, param in self.named_parameters():
                if "critic" not in name:
                    param.requires_grad = False

        # Store tensors into CPU for now
        self.device = torch.device("cpu")

        # Load pre-trained experts
        nav_ckpt = torch.load(nav_checkpoint_path, map_location=self.device)
        gaze_ckpt = torch.load(gaze_checkpoint_path, map_location=self.device)
        self.expert_nav_policy = ckpt_to_policy(nav_ckpt, observation_space)
        self.expert_gaze_policy = ckpt_to_policy(gaze_ckpt, observation_space)

        self.nav_rnn_hx, _, self.nav_prev_actions = get_blank_params(
            nav_ckpt["config"],
            self.expert_nav_policy,
            self.device,
            num_envs=num_environments,
        )
        self.gaze_rnn_hx, _, self.gaze_prev_actions = get_blank_params(
            gaze_ckpt["config"],
            self.expert_gaze_policy,
            self.device,
            num_envs=num_environments,
        )
        self.nav_action = None
        self.gaze_action = None
        self.num_experts = 2
        self._obs_batching_cache = ObservationBatchingCache()

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            nav_checkpoint_path=config.RL.POLICY.nav_checkpoint_path,
            gaze_checkpoint_path=config.RL.POLICY.gaze_checkpoint_path,
            goal_hidden_size=config.RL.PPO.get("goal_hidden_size", 0),
            fuse_states=config.RL.POLICY.fuse_states,
            num_environments=config.NUM_ENVIRONMENTS,
            hidden_size=config.RL.PPO.hidden_size,
            train_critic_only=config.RL.get("train_critic_only", False),
        )

    def to(self, device, *args):
        super().to(device, *args)
        self.nav_rnn_hx = self.nav_rnn_hx.to(device)
        self.gaze_rnn_hx = self.gaze_rnn_hx.to(device)
        self.nav_prev_actions = self.nav_prev_actions.to(device)
        self.gaze_prev_actions = self.gaze_prev_actions.to(device)
        self.expert_nav_policy = self.expert_nav_policy.to(device)
        self.expert_gaze_policy = self.expert_gaze_policy.to(device)
        self.device = device

    def transform_obs(self, observations, masks):
        """
        Inserts expert actions into the observations

        :param observations: list of dictionaries
        :param masks: torch tensor.bool
        :return: observations with expert actions
        """
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        masks_device = (
            masks.to(self.device) if masks.device != self.device else masks
        )
        with torch.no_grad():
            (
                _,
                self.nav_action,
                _,
                self.nav_rnn_hx,
            ) = self.expert_nav_policy.act(
                batch, self.nav_rnn_hx, self.nav_prev_actions, masks_device
            )
            (
                _,
                self.gaze_action,
                _,
                self.gaze_rnn_hx,
            ) = self.expert_gaze_policy.act(
                batch, self.gaze_rnn_hx, self.gaze_prev_actions, masks_device
            )
        self.nav_prev_actions.copy_(self.nav_action)
        self.gaze_prev_actions.copy_(self.gaze_action)

        # Move expert actions to CPU (for observation/action_arg insertion)
        self.nav_action = self.nav_action.detach().cpu()
        self.gaze_action = self.gaze_action.detach().cpu()

        # Insert into each observation; batch_obs loads these into gpu later
        num_envs = len(observations)
        for index_env in range(num_envs):
            observations[index_env][EXPERT_NAV_UUID] = self.nav_action[
                index_env
            ].numpy()
            observations[index_env][EXPERT_GAZE_UUID] = self.gaze_action[
                index_env
            ].numpy()

        return observations

    def action_to_dict(self, action, index_env, **kwargs):
        # Merge mixer's actions with experts'
        gaze_action = self.gaze_action[index_env]
        nav_action = self.nav_action[index_env]
        # experts_action = torch.cat([gaze_action, nav_action])
        # step_action = experts_action + action
        # step_action = experts_action
        step_action = action.to(torch.device("cpu"))

        # Add expert actions as action_args for reward calculation in RLEnv
        expert_args = {
            EXPERT_GAZE_UUID: gaze_action,
            EXPERT_NAV_UUID: nav_action,
        }
        step_data = {
            "action": {"action": step_action, **expert_args, **kwargs}
        }

        return step_data


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsMask(NavGazeMixtureOfExpertsRes):
    """
    This policy will signal which expert it thinks is most appropriate, use it
    to form a 'base' action composed of that expert's actions and 0s for the
    actions it does not control, and finally will also output actions that will
    be added to this base action.

    Unfortunately this causes the action space of the policy to be larger than
    that of the environment. We must ensure that RolloutStorage uses the
    policy's action space rather than the environment's, or else log_probs
    cannot be generated in the ppo.py script.
    """

    def __init__(self, residuals_on_inactive, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residuals_on_inactive = residuals_on_inactive
        self.nav_action_mask = None
        self.gaze_action_mask = None
        self.place_action_mask = None
        self.arm_action_mask = None

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        """Add actions for masking experts"""
        if config.RL.POLICY.get("place_checkpoint_path", "") == "":
            num_experts = 2
        else:
            num_experts = 3
        actual_action_space = Box(
            -1.0, 1.0, (action_space.shape[0] + num_experts,)
        )
        return cls(
            observation_space=observation_space,
            action_space=actual_action_space,
            nav_checkpoint_path=config.RL.POLICY.nav_checkpoint_path,
            gaze_checkpoint_path=config.RL.POLICY.gaze_checkpoint_path,
            goal_hidden_size=config.RL.PPO.get("goal_hidden_size", 0),
            fuse_states=config.RL.POLICY.fuse_states,
            num_environments=config.NUM_ENVIRONMENTS,
            hidden_size=config.RL.PPO.hidden_size,
            train_critic_only=config.RL.get("train_critic_only", False),
            residuals_on_inactive=config.RL.POLICY.residuals_on_inactive,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        """
        The residual actions need to be masked by the mask outputs in order for
        the log_probs to encourage the unused residuals to go towards zero if
        deemed a good decision, or away from zero if not.

        The last E actions are taken to be the mask actions, where E == # of
        experts. The mask order is Nav, Gaze, Place. The action order is Arm,
        Base.
        """
        value, action, action_log_probs, rnn_hidden_states = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic,
        )

        (
            residual_arm_actions,
            residual_base_actions,
            expert_masks,
        ) = torch.split(
            action, [ARM_ACTIONS, BASE_ACTIONS, self.num_experts], dim=1
        )

        # Zero out low residual values
        low_arm_residuals = (residual_arm_actions < 0.1) + (
            residual_arm_actions > -0.1
        )
        low_base_residuals = (residual_base_actions < 0.1) + (
            residual_base_actions > -0.1
        )
        residual_arm_actions[low_arm_residuals] = 0.0
        residual_base_actions[low_base_residuals] = 0.0

        # Generate arm and base action mask for later use (self.action_to_dict)
        self.get_action_masks(expert_masks)

        if self.residuals_on_inactive:
            # Residuals for unselected (inactive) experts are negated here
            residual_arm_actions = residual_arm_actions * (
                1.0 - self.arm_action_mask
            )
            residual_base_actions = residual_base_actions * (
                1.0 - self.nav_action_mask
            )

        action = torch.cat(
            [residual_arm_actions, residual_base_actions, expert_masks], dim=1
        )

        return value, action, action_log_probs, rnn_hidden_states

    def get_action_masks(self, expert_masks):
        activation_mask = torch.where(expert_masks > 0, 1.0, 0.0)
        activation_mask = torch.split(
            activation_mask, [1] * self.num_experts, dim=1
        )
        if self.num_experts == 2:
            self.nav_action_mask, self.gaze_action_mask = [
                m.repeat(1, num_actions)
                for m, num_actions in zip(
                    activation_mask, [BASE_ACTIONS, ARM_ACTIONS]
                )
            ]
            self.arm_action_mask = self.gaze_action_mask
        elif self.num_experts == 3:
            (
                self.nav_action_mask,
                self.gaze_action_mask,
                self.place_action_mask,
            ) = [
                m.repeat(1, num_actions)
                for m, num_actions in zip(
                    activation_mask, [BASE_ACTIONS, ARM_ACTIONS, ARM_ACTIONS]
                )
            ]
            # Arm mask is the union of the gaze and place masks
            self.arm_action_mask = torch.clip(
                self.gaze_action_mask + self.place_action_mask, 0.0, 1.0
            )
        else:
            raise NotImplementedError

    def action_to_dict(self, action, index_env, use_residuals=True, **kwargs):
        # Merge mixer's actions with experts'
        gaze_action = self.gaze_action[index_env]
        nav_action = self.nav_action[index_env]
        nav_action_mask = self.nav_action_mask[index_env].detach().cpu()
        gaze_action_mask = self.gaze_action_mask[index_env].detach().cpu()

        # Compile an action based on the actions of the selected experts
        base_action = nav_action * nav_action_mask
        if self.num_experts == 2:
            arm_action = gaze_action * gaze_action_mask
        elif self.num_experts == 3:
            # TODO: both place and pick cannot be used at the same time...
            raise NotImplementedError
        else:
            raise NotImplementedError
        experts_action = torch.cat([arm_action, base_action])

        if use_residuals:
            residual_action = action[: ARM_ACTIONS + BASE_ACTIONS]
            step_action = experts_action + residual_action
        else:
            step_action = experts_action

        # Add expert actions as action_args for reward calculation in RLEnv
        expert_args = {
            EXPERT_GAZE_UUID: gaze_action,
            EXPERT_NAV_UUID: nav_action,
            EXPERT_MASKS_UUID: action[ARM_ACTIONS + BASE_ACTIONS :],
        }
        step_data = {
            "action": {"action": step_action, **expert_args, **kwargs}
        }

        return step_data
