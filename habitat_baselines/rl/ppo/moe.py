from habitat_baselines.rl.ppo.policy import (
    Policy,
    PointNavBaselineNet,
    PointNavBaselinePolicy,
)

import torch
from gym import spaces

from habitat.config import Config

from habitat_baselines.common.baseline_registry import baseline_registry

from gym.spaces import Box, Dict
import numpy as np


"""
Focus first on NavGaze MoE....
"""


@baseline_registry.register_policy
class NavGazeMixtureOfExperts(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        nav_checkpoint_path,
        gaze_checkpoint_path,
        goal_hidden_size,
        fuse_states,
        num_environments,
        hidden_size: int = 512,
        **kwargs,
    ):
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

        # Load navigation expert
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        print(nav_checkpoint_path)
        exit()
        nav_checkpoint = torch.load(nav_checkpoint_path)
        nav_config = nav_checkpoint["config"]

        # We need to remove the arm depth camera from navigation
        nav_obs_space = Dict(observation_space.spaces.copy())
        nav_obs_space.spaces.pop("arm_depth")
        self.expert_nav_policy = PointNavBaselinePolicy.from_config(
            config=nav_config,
            observation_space=nav_obs_space,
            action_space=Box(shape=(2,), low=-1, high=1, dtype=np.float32),
        )

        data_dict = nav_checkpoint["state_dict"]
        self.expert_nav_policy.load_state_dict(
            {
                k[len("actor_critic.") :]: torch.tensor(v)
                for k, v in data_dict.items()
                if k.startswith("actor_critic.")
            }
        )

        # Load gaze expert
        gaze_checkpoint = torch.load(gaze_checkpoint_path)
        self.expert_gaze_policy = PointNavBaselinePolicy.from_config(
            config=gaze_checkpoint["config"],
            observation_space=observation_space,
            action_space=Box(shape=(8,), low=-1, high=1, dtype=np.float32),
        )
        data_dict = gaze_checkpoint["state_dict"]
        self.expert_gaze_policy.load_state_dict(
            {
                k[len("actor_critic.") :]: torch.tensor(v)
                for k, v in data_dict.items()
                if k.startswith("actor_critic.")
            }
        )

        # Need to keep track of hidden states and prev_actions for each expert
        self.nav_rnn_hidden_states = torch.zeros(
            num_environments,
            1,  # num_recurrent_layers
            512,  # ppo_cfg.hidden_size,
            dtype=torch.float32,
        )
        self.gaze_rnn_hidden_states = torch.zeros(
            num_environments,
            1,  # num_recurrent_layers
            512,  # ppo_cfg.hidden_size,
            dtype=torch.float32,
        )
        self.nav_prev_actions = torch.zeros(
            num_environments,
            2,
            dtype=torch.float32,
        )
        self.gaze_prev_actions = torch.zeros(
            num_environments,
            8,
            dtype=torch.float32,  # ppo_cfg.hidden_size,
        )
        self.nav_action = None
        self.gaze_action = None

    # Overload .to() method
    def to(self, device, *args):
        super().to(device, *args)
        self.nav_rnn_hidden_states = self.nav_rnn_hidden_states.to(device)
        self.gaze_rnn_hidden_states = self.gaze_rnn_hidden_states.to(device)
        self.nav_prev_actions = self.nav_prev_actions.to(device)
        self.gaze_prev_actions = self.gaze_prev_actions.to(device)
        self.expert_nav_policy = self.expert_nav_policy.to(device)
        self.expert_gaze_policy = self.expert_gaze_policy.to(device)

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        """
        Action space needs to be overwritten, as by default it is extracted from the
        env. Anticipate necessary changes to prev_actions and RolloutStorage.

        We need an action for each expert. Above 0 means execute. Below means don't.
        """

        action_space = Box(-1.0, 1.0, (2,))

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            nav_checkpoint_path=config.RL.POLICY.nav_checkpoint_path,
            gaze_checkpoint_path=config.RL.POLICY.gaze_checkpoint_path,
            goal_hidden_size=config.RL.PPO.get("goal_hidden_size", 0),
            fuse_states=config.RL.POLICY.fuse_states,
            num_environments=config.NUM_ENVIRONMENTS,
            hidden_size=config.RL.PPO.hidden_size,
        )

    """
    act() needs to be changed so that it actually passes the observations into the
    experts and gets their actions and updates their hidden states.
    """

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        value, action, action_log_probs, rnn_hidden_states = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
        )

        (
            _,
            self.nav_action,
            _,
            self.nav_rnn_hidden_states,
        ) = self.expert_nav_policy.act(
            observations,
            self.nav_rnn_hidden_states,
            self.nav_prev_actions,
            # masks,
            torch.ones_like(masks, device='cuda:0'),
            deterministic=True,
        )
        # print("observations[0]['obj_start_sensor']", observations[0]['obj_start_sensor'])
        # print('self.nav_action', self.nav_action)
        # exit()
        self.nav_prev_actions.copy_(self.nav_action)

        (
            _,
            self.gaze_action,
            _,
            self.gaze_rnn_hidden_states,
        ) = self.expert_gaze_policy.act(
            observations,
            self.gaze_rnn_hidden_states,
            self.gaze_prev_actions,
            masks,
            deterministic=True,
        )
        self.gaze_prev_actions.copy_(self.gaze_action)

        return value, action, action_log_probs, rnn_hidden_states

    def choose_mix_of_actions(self, actions):
        """
        Given a NavGaze policy, reformat the MoE action to actually incorporate expert
        actions

        :param actions:
        :return: step_data, similar to below
        [{'action': array([-1.360452  ,  1.7130293 , -0.706296  , -1.5175911 , -2.1926935 ,
           -0.96006346], dtype=float32)}]

        arm_ac = action[0][:3].cpu().numpy()
        grip_ac = None if not self.place_success else 0
        base_ac = action[0][-2:].cpu().numpy()
        step_data = [{'action': a.numpy()} for a in actions.to(device="cpu")]
        """

        step_data = []
        for mix_action, base_action, gaze_action in zip(
            actions.to(device="cpu"),
            self.nav_action.to(device="cpu"),
            self.gaze_action.to(device="cpu"),
        ):
            use_nav, use_gaze = mix_action.numpy()
            if use_nav > 0.0:
                base_step_action = base_action.numpy()
            else:
                base_step_action = np.zeros(2, dtype=np.float32)
            if use_gaze > 0.0:
                gaze_step_action = gaze_action.numpy()
            else:
                gaze_step_action = np.zeros(8, dtype=np.float32)

            step_data.append(
                {
                    "action": np.concatenate(
                        [gaze_step_action, base_step_action]
                    )
                }
            )

        return step_data
    #
    # def choose_single_mix_of_actions(self, action):
    #     """
    #     Given a NavGaze policy, reformat the MoE action to actually incorporate expert
    #     actions
    #
    #     :param actions:
    #     :return: step_data, similar to below
    #     [{'action': array([-1.360452  ,  1.7130293 , -0.706296  , -1.5175911 , -2.1926935 ,
    #        -0.96006346], dtype=float32)}]
    #
    #     arm_ac = action[0][:3].cpu().numpy()
    #     grip_ac = None if not self.place_success else 0
    #     base_ac = action[0][-2:].cpu().numpy()
    #     step_data = [{'action': a.numpy()} for a in actions.to(device="cpu")]
    #     """
    #
    #     use_nav, use_gaze = action.cpu().numpy()
    #     if use_nav > 0.0:
    #         base_step_action = base_action.numpy()
    #     else:
    #         base_step_action = np.zeros(2, dtype=np.float32)
    #     if use_gaze > 0.0:
    #         gaze_step_action = gaze_action.numpy()
    #     else:
    #         gaze_step_action = np.zeros(8, dtype=np.float32)
    #
    #     return {"action": np.concatenate([base_step_action, gaze_step_action])}
