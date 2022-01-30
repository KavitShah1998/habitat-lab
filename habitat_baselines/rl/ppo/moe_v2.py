#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
from torch import nn as nn

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import GaussianNet, initialized_linear


def construct_mlp_base(input_size, hidden_size, num_layers=3):
    """Returns 3-layer MLP as a list of layers"""
    layers = []
    prev_size = input_size
    for _ in range(num_layers):
        layers.append(
            initialized_linear(prev_size, hidden_size, gain=np.sqrt(2))
        )
        layers.append(nn.ReLU())
        prev_size = hidden_size
    return layers


class MoePolicy(Policy, nn.Module):
    """
    Need 3 networks:
    A Net->Gaussian Gating Network (action shape == num_experts)
    A Net->Gaussian Residual Network (action shape == 2 base + 4 joint actions)
    A Net->Critic Network (used for RL training, unused for BC or test time)
    """

    def __init__(self, observation_space, fuse_states, num_gates, num_actions):
        nn.Module.__init__(self)
        hidden_size = 512
        self.num_gates = num_gates
        self.num_actions = num_actions
        self.fuse_states = fuse_states
        input_size = sum(
            [observation_space.spaces[n].shape[0] for n in self.fuse_states]
        )

        # Gating actor
        self.gating_actor = nn.Sequential(
            *construct_mlp_base(input_size, hidden_size),
            GaussianNet(hidden_size, num_gates),
        )

        # Residual actor
        self.residual_actor = nn.Sequential(
            *construct_mlp_base(input_size, hidden_size),
            GaussianNet(hidden_size, num_actions),
        )

        # Critic
        self.critic = nn.Sequential(
            *construct_mlp_base(input_size, hidden_size),
            initialized_linear(hidden_size, 1, gain=np.sqrt(2)),
        )

    def obs_to_tensor(self, observations):
        # Convert double to float if found
        for k, v in observations.items():
            if v.dtype is torch.float64:
                observations[k] = v.type(torch.float32)
        return torch.cat([observations[k] for k in self.fuse_states], dim=1)

    def act(
        self,
        observations,
        rnn_hidden_states,  # don't use RNNs for now
        prev_actions,  # don't use prev_actions for now
        masks,  # don't use RNNs for now
        deterministic=False,
    ):
        (
            gating_distribution,
            residual_distribution,
            value,
        ) = self.compute_actions_and_value(observations)

        action_and_log_probs = []
        for d in [gating_distribution, residual_distribution]:
            act = d.mode() if deterministic else d.sample()
            log_probs = d.log_probs(act)
            action_and_log_probs.extend([act, log_probs])
        gate_act, gate_log_p, res_act, res_log_p = action_and_log_probs

        action = torch.cat([gate_act, res_act], dim=1)
        action_log_probs = gate_log_p + res_log_p

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        return self.critic(self.obs_to_tensor(observations))

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        (
            gating_distribution,
            residual_distribution,
            value,
        ) = self.compute_actions_and_value(observations)

        gate_act, res_act = torch.split(
            action, [self.num_gates, self.num_actions], dim=1
        )

        action_log_probs = gating_distribution.log_probs(
            gate_act
        ) + residual_distribution.log_probs(res_act)
        distribution_entropy = (
            gating_distribution.entropy() + residual_distribution.entropy()
        )

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    def compute_actions_and_value(self, observations):
        observations_tensor = self.obs_to_tensor(observations)
        gating_distribution = self.gating_actor(observations_tensor)
        residual_distribution = self.residual_actor(observations_tensor)
        value = self.critic(observations_tensor)

        return gating_distribution, residual_distribution, value

    def forward(self, *x):
        raise NotImplementedError
