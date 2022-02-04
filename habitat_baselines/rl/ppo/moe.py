import numpy as np
import torch
from gym.spaces import Box, Dict

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.moe_v2 import MoePolicy
from habitat_baselines.rl.ppo.policy import PointNavBaselineNet, Policy
from habitat_baselines.rl.ppo.sequential import (
    ckpt_to_policy,
    get_blank_params,
)
from habitat_baselines.utils.common import ObservationBatchingCache, batch_obs

ARM_ACTIONS = 4  # 4 controllable joints
BASE_ACTIONS = 2  # linear and angular vel
EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"
EXPERT_MASKS_UUID = "expert_masks"
VISUAL_FEATURES_UUID = "visual_features"


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsRes(MoePolicy):
    def __init__(
        self,
        observation_space: Dict,
        action_space,
        nav_checkpoint_path,
        gaze_checkpoint_path,
        goal_hidden_size,
        fuse_states,
        num_environments,
        freeze_keys,
        hidden_size: int = 512,
        force_blind=False,
        *args,
        **kwargs,
    ):
        # Determines if we use visual observations or features from experts
        self.blind = force_blind
        observation_space_copy = Dict(observation_space.spaces.copy())
        if self.blind:
            vis_keys = [
                k for k in observation_space.spaces.keys() if "depth" in k
            ]
            for k in vis_keys:
                observation_space.spaces.pop(k)
            # Add visual features to the observation space
            observation_space.spaces[VISUAL_FEATURES_UUID] = Box(
                -np.inf, np.inf, (hidden_size * 2,)
            )
            fuse_states = [VISUAL_FEATURES_UUID] + fuse_states

        # Add expert actions to the observation space and fuse_states
        observation_space.spaces[EXPERT_NAV_UUID] = Box(
            -1.0, 1.0, (BASE_ACTIONS,)
        )
        observation_space.spaces[EXPERT_GAZE_UUID] = Box(
            -1.0, 1.0, (ARM_ACTIONS,)
        )
        fuse_states.extend([EXPERT_NAV_UUID, EXPERT_GAZE_UUID])

        # Instantiate MoE's own policy
        # TODO: don't hardcode gate amount
        super().__init__(
            observation_space,
            fuse_states=fuse_states,
            num_gates=2,
            num_actions=BASE_ACTIONS + ARM_ACTIONS,
        )

        # For RolloutStorage in ppo_trainer.py
        self.policy_action_space = action_space

        # Get model params before experts get loaded in
        self.model_params = [p for p in self.parameters() if p.requires_grad]

        # Freeze weights that contain any freeze keys
        for key in freeze_keys:
            for name, param in self.named_parameters():
                if key in name:
                    param.requires_grad = False
                    print("Freezing:", name)

        # Store tensors into CPU for now
        self.device = torch.device("cpu")

        # Load pre-trained experts
        nav_ckpt = torch.load(nav_checkpoint_path, map_location=self.device)
        gaze_ckpt = torch.load(gaze_checkpoint_path, map_location=self.device)
        self.expert_nav_policy = ckpt_to_policy(
            nav_ckpt, observation_space_copy
        )
        self.expert_gaze_policy = ckpt_to_policy(
            gaze_ckpt, observation_space_copy
        )

        # Freeze expert weights
        for name, param in self.named_parameters():
            if "expert" in name:
                param.requires_grad = False

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
        self.nav_masks = torch.ones(num_environments, 1, dtype=torch.bool)
        self.gaze_masks = torch.ones(num_environments, 1, dtype=torch.bool)
        self.prev_nav_masks = torch.zeros(
            num_environments, 1, dtype=torch.bool
        )
        self.prev_gaze_masks = torch.zeros(
            num_environments, 1, dtype=torch.bool
        )
        self.nav_action = None
        self.gaze_action = None
        self.num_experts = 2  # TODO
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
            force_blind=config.RL.POLICY.force_blind,
            freeze_keys=config.RL.POLICY.freeze_keys,
        )

    def to(self, device, *args):
        super().to(device, *args)
        self.nav_rnn_hx = self.nav_rnn_hx.to(device)
        self.gaze_rnn_hx = self.gaze_rnn_hx.to(device)
        self.nav_prev_actions = self.nav_prev_actions.to(device)
        self.gaze_prev_actions = self.gaze_prev_actions.to(device)
        self.expert_nav_policy = self.expert_nav_policy.to(device)
        self.expert_gaze_policy = self.expert_gaze_policy.to(device)
        self.nav_masks = self.nav_masks.to(device)
        self.gaze_masks = self.gaze_masks.to(device)
        self.prev_nav_masks = self.prev_nav_masks.to(device)
        self.prev_gaze_masks = self.prev_gaze_masks.to(device)
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
        nav_masks = torch.logical_and(masks_device, self.nav_masks)
        gaze_masks = torch.logical_and(masks_device, self.gaze_masks)
        with torch.no_grad():
            (
                _,
                self.nav_action,
                _,
                self.nav_rnn_hx,
            ) = self.expert_nav_policy.act(
                batch, self.nav_rnn_hx, self.nav_prev_actions, nav_masks
            )
            (
                _,
                self.gaze_action,
                _,
                self.gaze_rnn_hx,
            ) = self.expert_gaze_policy.act(
                batch, self.gaze_rnn_hx, self.gaze_prev_actions, gaze_masks
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

            if self.blind:
                # Remove visual observations
                vis_keys = [
                    k for k in observations[index_env].keys() if "depth" in k
                ]
                for k in vis_keys:
                    observations[index_env].pop(k)

                # Stitch and add visual features from the experts
                visual_features = [
                    p.net.pred_visual_features[index_env]
                    for p in [self.expert_nav_policy, self.expert_gaze_policy]
                ]
                observations[index_env][VISUAL_FEATURES_UUID] = (
                    torch.cat(visual_features).cpu().numpy()
                )

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

    def __init__(self, residuals_on_inactive, use_residuals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residuals_on_inactive = residuals_on_inactive
        self.use_residuals = use_residuals
        self.nav_action_mask = None
        self.gaze_action_mask = None
        self.place_action_mask = None
        self.arm_action_mask = None
        self.num_masks = self.num_experts

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
            use_residuals=config.RL.POLICY.use_residuals,
            force_blind=config.RL.POLICY.force_blind,
            freeze_keys=config.RL.POLICY.freeze_keys,
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
            action, [ARM_ACTIONS, BASE_ACTIONS, self.num_masks], dim=1
        )

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

        def reset_hx(prev, curr):
            prev_bool, curr_bool = torch.eq(prev, 1.0), torch.eq(curr, 1.0)
            reset = torch.logical_and(torch.logical_not(prev_bool), curr_bool)
            return torch.logical_not(reset)

        if self.num_experts == 2:
            nav_masks, gaze_masks = activation_mask
            self.nav_action_mask, self.gaze_action_mask = [
                m.repeat(1, num_actions)
                for m, num_actions in zip(
                    activation_mask, [BASE_ACTIONS, ARM_ACTIONS]
                )
            ]
            self.arm_action_mask = self.gaze_action_mask
        elif self.num_experts == 3:
            nav_masks, gaze_masks, place_masks = activation_mask
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

        # Zero out mask values indicating reselection of an expert
        self.nav_masks = reset_hx(self.prev_nav_masks, nav_masks)
        self.gaze_masks = reset_hx(self.prev_gaze_masks, gaze_masks)
        if self.num_experts == 3:
            raise NotImplementedError  # TODO
        self.prev_nav_masks = nav_masks
        self.prev_gaze_masks = gaze_masks

    def action_to_dict(self, action, index_env, use_residuals=True, **kwargs):
        if self.use_residuals is not None:
            use_residuals = self.use_residuals

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
        masks_arg = action[ARM_ACTIONS + BASE_ACTIONS :].detach().cpu().numpy()
        expert_args = {
            EXPERT_GAZE_UUID: gaze_action,
            EXPERT_NAV_UUID: nav_action,
            EXPERT_MASKS_UUID: masks_arg,
        }
        step_data = {
            "action": {"action": step_action, **expert_args, **kwargs}
        }

        return step_data


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsMaskSingle(NavGazeMixtureOfExpertsMask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_masks = 1

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        actual_action_space = Box(-1.0, 1.0, (action_space.shape[0] + 1,))

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
            use_residuals=config.RL.POLICY.use_residuals,
            force_blind=config.RL.POLICY.force_blind,
        )

    def get_action_masks(self, expert_masks):
        if self.num_experts == 2:
            nav_masks = torch.where(expert_masks < 0, 1.0, -1.0)
            gaze_masks = torch.where(expert_masks > 0, 1.0, -1.0)
            expert_masks = [nav_masks, gaze_masks]
        elif self.num_experts == 3:
            nav_masks = torch.where(expert_masks < -0.25, 1.0, -1.0)
            gaze_masks = torch.where(
                torch.logical_and(-0.25 < expert_masks, expert_masks < 0.25),
                1.0,
                -1.0,
            )
            place_masks = torch.where(expert_masks > 0.25, 1.0, -1.0)
            expert_masks = [nav_masks, gaze_masks, place_masks]
        else:
            raise NotImplementedError
        expert_masks = torch.cat(expert_masks, dim=1)
        super().get_action_masks(expert_masks)


@baseline_registry.register_policy
class NavGazeMixtureOfExpertsMaskCombo(NavGazeMixtureOfExpertsMask):
    """
    The difference between Combo and Single is that the final action, the
    experts mask, is not the output of a Gaussian action distribution (-1 - 1),
    but rather is the output of a Categorical action distribution (0 - 2) or
    (0 - 4)
    """

    def __init__(self, num_combos, freeze_residuals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_masks = 1
        self.num_combos = num_combos
        self.freeze_residuals = freeze_residuals
        if freeze_residuals:
            for name, param in self.named_parameters():
                if "gaussian_net" in name:
                    param.requires_grad = False

        """Nav, Gaze, NavGaze | Place, NavPlace"""
        filter_valid_combos = lambda x: [i for i in x if i < num_combos]
        self.nav_combo_ids = filter_valid_combos([0, 2, 4])
        self.gaze_combo_ids = filter_valid_combos([1, 2])
        self.place_combo_ids = filter_valid_combos([3, 4])

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        num_actions = action_space.shape[0]
        actual_action_space = Box(-1.0, 1.0, (num_actions + 1,))

        num_combos = config.RL.POLICY.num_combos
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
            force_blind=config.RL.POLICY.force_blind,
            residuals_on_inactive=config.RL.POLICY.residuals_on_inactive,
            num_combos=num_combos,
            freeze_residuals=config.RL.POLICY.freeze_residuals,
            # For GaussianCategorical action distribution
            gaussian_categorical=True,
            num_outputs_gaussian=num_actions,
            num_outputs_categorical=num_combos,
        )

    def get_action_masks(self, expert_masks):
        cmbs = [self.nav_combo_ids, self.gaze_combo_ids, self.place_combo_ids]

        expert_masks_out = []
        for combo_ids in cmbs:
            if len(combo_ids) == 0:
                continue
            expert_mask = torch.zeros_like(expert_masks, dtype=torch.bool)
            for combo_id in combo_ids:
                combo_id_mask = expert_masks == combo_id
                expert_mask = torch.logical_or(combo_id_mask, expert_mask)
            expert_mask_out = torch.where(expert_mask, 1.0, -1.0)
            expert_masks_out.append(expert_mask_out)

        expert_masks_out = torch.cat(expert_masks_out, dim=1)

        super().get_action_masks(expert_masks_out)

    def action_to_dict(self, action, index_env, **kwargs):
        return super().action_to_dict(
            action,
            index_env,
            use_residuals=not self.freeze_residuals,
            **kwargs,
        )
