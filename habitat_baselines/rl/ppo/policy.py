#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
from gym import spaces
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
from habitat_baselines.utils.common import CategoricalNet

# import rlf.policies.utils as putils
# import rlf.rl.utils as rutils
from habitat.core.spaces import ActionSpace

"""
At minimum, need to support:
- depth (yes)
- depth+bbox (yes)
- head-depth (yes)
- head-depth+depth+bbox (NO!!!)
"""


ARM_VISION_KEYS = ["arm_depth", "arm_rgb", "arm_depth_bbox"]
HEAD_VISION_KEYS = ["depth", "rgb"]

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions
).sum(-1, keepdim=True)
normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        def weight_init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module

        init_ = lambda m: weight_init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        action_logstd = self.logstd.expand_as(action_mean)
        return FixedNormal(action_mean, action_logstd.exp())


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, action_space):
        super().__init__()
        self.net = net

        if net is not None:
            if isinstance(action_space, ActionSpace):
                self.action_distribution = CategoricalNet(
                    self.net.output_size, action_space.n
                )
            else:
                self.action_distribution = DiagGaussian(
                    self.net.output_size, action_space.shape[0]
                )

            self.critic = CriticHead(self.net.output_size)

        self.count = 0

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
        # print('batched_obs["depth"]', observations["depth"])
        # print(
        #     'batched_obs["target_point_goal_gps_and_compass_sensor"]',
        #     observations["target_point_goal_gps_and_compass_sensor"],
        # )
        # print('self.hidden_state', rnn_hidden_states)
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

        # print('self.prev_actions', prev_actions)
        # print('self.masks', masks)
        # print('action', action)
        # self.count += 1
        # if self.count == 2:
        #     exit()
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
        # print(observation_space)
        # print(action_space)
        # print(hidden_size)
        # print(kwargs)
        # [print(9) for _ in range(30)]
        # exit()
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
        # [print(config.RL.POLICY.fuse_states) for _ in range(30)]
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

        # print(observation_space.spaces.keys())
        # exit()

        if (
            "target_point_goal_gps_and_compass_sensor"
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                "target_point_goal_gps_and_compass_sensor"
            ].shape[0]
        elif (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
        elif PointGoalSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
        elif ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(
                goal_observation_space, hidden_size, False
            )
            self._n_input_goal = hidden_size

        self.fuse_states = fuse_states
        self._n_input_goal = sum(
            [observation_space.spaces[n].shape[0] for n in self.fuse_states]
        )

        head_visual_inputs = 0
        arm_visual_inputs = 0
        for obs_key in observation_space.spaces.keys():
            if obs_key in ARM_VISION_KEYS:
                arm_visual_inputs += 1
            elif obs_key in HEAD_VISION_KEYS:
                head_visual_inputs += 1
        self.num_cnns = min(1, head_visual_inputs) + min(1, arm_visual_inputs)

        self._hidden_size = hidden_size

        if self.num_cnns == 1:
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

        state_dim = self._n_input_goal
        self._goal_hidden_size = goal_hidden_size
        if self._goal_hidden_size != 0:
            self.goal_encoder = nn.Sequential(
                nn.Linear(self._n_input_goal, self._goal_hidden_size),
                nn.ReLU(),
                nn.Linear(self._goal_hidden_size, self._goal_hidden_size),
                nn.ReLU(),
            )
            state_dim = self._goal_hidden_size

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else hidden_size * self.num_cnns) + state_dim,
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
        target_encoding = None
        if "target_point_goal_gps_and_compass_sensor" in observations:
            target_encoding = observations[
                "target_point_goal_gps_and_compass_sensor"
            ]
        elif IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            target_encoding = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
        elif PointGoalSensor.cls_uuid in observations:
            target_encoding = observations[PointGoalSensor.cls_uuid]
        elif ImageGoalSensor.cls_uuid in observations:
            image_goal = observations[ImageGoalSensor.cls_uuid]
            target_encoding = self.goal_visual_encoder({"rgb": image_goal})

        if len(self.fuse_states) > 0:
            target_encoding = torch.cat(
                [observations[k] for k in self.fuse_states], dim=-1
            )

        if target_encoding is None:
            x = []
        else:
            x = [target_encoding]
        # print('target_encoding', target_encoding)

        if self._goal_hidden_size != 0:
            x = self.goal_encoder(torch.cat(x, dim=1))

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.num_cnns == 1:
                if self._goal_hidden_size == 0:
                    x = [perception_embed] + x
                else:
                    x = [perception_embed, x]
            elif self.num_cnns == 2:
                perception_embed_2 = self.visual_encoder2(observations)
                if self._goal_hidden_size == 0:
                    x = [perception_embed, perception_embed_2] + x
                else:
                    x = [perception_embed, perception_embed_2, x]

        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )

        return x_out, rnn_hidden_states
