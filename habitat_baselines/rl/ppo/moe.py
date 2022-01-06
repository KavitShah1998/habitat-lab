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

GAZE_ACTIONS = 4  # 4 controllable joints
NAV_ACTIONS = 2  # linear and angular vel
EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"


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
            -1.0, 1.0, (NAV_ACTIONS,)
        )
        observation_space.spaces[EXPERT_GAZE_UUID] = Box(
            -1.0, 1.0, (GAZE_ACTIONS,)
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
        self._obs_batching_cache = ObservationBatchingCache()

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Dict, action_space
    ):
        """
        Action space needs to be overwritten, as by default it is extracted from the
        env. Anticipate necessary changes to prev_actions and RolloutStorage.

        We need an action for each expert. Above 0 means execute. Below means don't.
        """

        action_space = Box(-1.0, 1.0, (GAZE_ACTIONS + NAV_ACTIONS,))

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            nav_checkpoint_path=config.RL.POLICY.nav_checkpoint_path,
            gaze_checkpoint_path=config.RL.POLICY.gaze_checkpoint_path,
            goal_hidden_size=config.RL.PPO.get("goal_hidden_size", 0),
            fuse_states=config.RL.POLICY.fuse_states,
            num_environments=config.NUM_ENVIRONMENTS,
            hidden_size=config.RL.PPO.hidden_size,
            train_critic_only=config.RL.get("train_critic_only", False)
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
