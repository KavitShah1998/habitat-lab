#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

from gym import spaces
import numpy as np
import torch
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import (
    CategoricalNet,
    GaussianNet,
    GaussianCategoricalNet,
    initialized_linear,
)

from habitat.core.spaces import ActionSpace


ARM_VISION_KEYS = ["arm_depth", "arm_rgb", "arm_depth_bbox"]
HEAD_VISION_KEYS = ["depth", "rgb"]


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self, net, action_space, gaussian_categorical=False, **kwargs
    ):
        super().__init__()
        self.net = net

        if net is not None:
            if isinstance(action_space, ActionSpace):
                self.action_distribution = CategoricalNet(
                    self.net.output_size, action_space.n
                )
            else:
                if gaussian_categorical:
                    self.action_distribution = GaussianCategoricalNet(
                        self.net.output_size, **kwargs
                    )
                else:
                    self.action_distribution = GaussianNet(
                        self.net.output_size, action_space.shape[0]
                    )

            self.critic = CriticHead(self.net.output_size)

        self.distribution = None

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        # Save for use in behavioral cloning the mean and std
        self.distribution = distribution

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        **kwargs,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        goal_hidden_size = config.RL.PPO.get("goal_hidden_size", 0)
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            goal_hidden_size=goal_hidden_size,
            fuse_states=config.RL.POLICY.fuse_states,
            force_blind=config.RL.POLICY.force_blind,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        goal_hidden_size,
        fuse_states,
        force_blind,
    ):
        super().__init__()

        self.fuse_states = fuse_states
        self._n_input_goal = sum(
            [observation_space.spaces[n].shape[0] for n in self.fuse_states]
        )

        # Construct CNNs
        head_visual_inputs = 0
        arm_visual_inputs = 0
        for obs_key in observation_space.spaces.keys():
            if obs_key in ARM_VISION_KEYS:
                arm_visual_inputs += 1
            elif obs_key in HEAD_VISION_KEYS:
                head_visual_inputs += 1
        self.num_cnns = min(1, head_visual_inputs) + min(1, arm_visual_inputs)

        self._hidden_size = hidden_size

        if self.num_cnns <= 1:
            if self.num_cnns == 0:
                force_blind = True
            self.visual_encoder = SimpleCNN(
                observation_space, hidden_size, force_blind
            )
        elif self.num_cnns == 2:
            # We are using both cameras; make a CNN for each
            head_obs_space, arm_obs_space = [
                spaces.Dict(
                    {
                        k: v
                        for k, v in observation_space.spaces.items()
                        if k not in obs_key_blacklist
                    }
                )
                for obs_key_blacklist in [ARM_VISION_KEYS, HEAD_VISION_KEYS]
            ]
            # Head CNN
            self.visual_encoder = SimpleCNN(
                head_obs_space, hidden_size, force_blind, head_only=True
            )
            # Arm CNN
            self.visual_encoder2 = SimpleCNN(
                arm_obs_space, hidden_size, force_blind, arm_only=True
            )
        else:
            raise RuntimeError(
                f"Only supports 1 or 2 CNNs not {self.num_cnns}"
            )

        # 2-layer MLP for non-visual inputs
        self._goal_hidden_size = goal_hidden_size
        if self._goal_hidden_size != 0:
            self.goal_encoder = nn.Sequential(
                initialized_linear(
                    self._n_input_goal, self._goal_hidden_size, gain=np.sqrt(2)
                ),
                nn.ReLU(),
                initialized_linear(
                    self._goal_hidden_size,
                    self._goal_hidden_size,
                    gain=np.sqrt(2),
                ),
                nn.ReLU(),
            )

        # Final RNN layer
        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else hidden_size * self.num_cnns)
            + self._goal_hidden_size,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        # Convert double to float if found
        for k, v in observations.items():
            if v.dtype is torch.float64:
                observations[k] = v.type(torch.float32)

        x = []

        # Visual observations
        if not self.is_blind:
            x.append(self.visual_encoder(observations))
            if self.num_cnns == 2:
                x.append(self.visual_encoder2(observations))

        # Non-visual observations
        if len(self.fuse_states) > 0:
            non_vis_obs = [observations[k] for k in self.fuse_states]
            x.append(self.goal_encoder(torch.cat(non_vis_obs, dim=-1)))

        # Final RNN layer
        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )

        return x_out, rnn_hidden_states
