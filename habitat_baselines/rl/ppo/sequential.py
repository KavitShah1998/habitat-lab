from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy, Policy

from collections import OrderedDict
import torch
from gym import spaces

from habitat.config import Config

from habitat_baselines.common.baseline_registry import baseline_registry

from gym.spaces import Box, Dict
import numpy as np


"""
Assumption: we only deal with nav, pick, place, navpick, and navplace

Inputs: policy checkpoints, including what kind of env they are for.

- Need a function that decides who is next when one policy declares it's done
- How does each policy declare that it is done? Depends on:
    - Policy type (easy)
    - Timeout (easy)
    - State of the environment (hard)
    -
- Completely disregard prev_action and hnn input
    - use the one saved from last time
- Need a way to update the target
    - This may become a bigger problem later

Agent starts either in nav or manipulation. Nav seems better.

Master observation and action spaces are decided by the

"""


def env_to_policy(config, state_dict, observation_space):
    env_name = config.hab_env_config

    # For navpick or navplace, we just feed the entire observation space
    # (both cameras + fuse_states)
    if env_name in ["navpick", "navplace"]:
        policy = PointNavBaselinePolicy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=Box(-1.0, 1.0, (6,)),
        )

    # For navigation, we get rid of the arm camera (wasn't used in training)
    else:
        raise NotImplementedError(f"env_name {env_name} not supported yet")

    policy.load_state_dict(
        {
            k[len("actor_critic.") :]: torch.tensor(v)
            for k, v in state_dict.items()
            if k.startswith("actor_critic.")
        }
    )

    return policy


def get_blank_params(skill, device):
    config = skill["config"]
    hidden_state = torch.zeros(
        1,  # Number of environments. Just do one.
        1,  # num_recurrent_layers. SimpleCNN uses 1.
        config.RL.PPO.hidden_size,  # ppo_cfg.hidden_size,
        device=device,
    )

    masks = torch.ones(
        1,  # Number of environments. Just do one.
        1,  # Just need one boolean.
        dtype=torch.bool,
        device=device,
    )

    policy = skill["policy"]
    num_actions = policy.action_distribution.fc_mean.out_features
    prev_actions = torch.zeros(
        1,  # Number of environments. Just do one.
        num_actions,
        device=device,
    )

    return hidden_state, masks, prev_actions


from torch import nn
# For evaluation only!
@baseline_registry.register_policy
class SequentialExperts(PointNavBaselinePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        experts,  # paths to .pth checkpoints
    ):
        # We just need this so things don't break...
        super().__init__(
            observation_space,
            action_space,
            hidden_size=512,
            goal_hidden_size=512,
            fuse_states=[],
            force_blind=False,
        )

        # Maps expert type (name of env used to train) to policies
        self.expert_skills = OrderedDict()
        for e in experts:
            checkpoint = torch.load(e, map_location="cpu")
            config = checkpoint["config"]
            skill_type = config.hab_env_config

            self.expert_skills[skill_type] = {}
            self.expert_skills[skill_type]["policy"] = env_to_policy(
                config,
                checkpoint["state_dict"],
                observation_space,
            )
            self.expert_skills[skill_type]["config"] = config

            # May seem redundant, but useful later
            self.expert_skills[skill_type]["skill_type"] = skill_type

        # Load things to CPU for now
        self.device = torch.device("cpu")

        # Assume first checkpoint given corresponds to the first policy to use
        self.current_skill = list(self.expert_skills.values())[0]
        self.hidden_state, self.masks, self.prev_actions = get_blank_params(
            self.current_skill,
            self.device,
        )

        self.num_steps = 0
        self.num_transitions = 0
        self.next_skill_type = ""

    def reset(self):
        print("Resetting SequentialExperts...")
        self.current_skill = list(self.expert_skills.values())[0]
        self.hidden_state, self.masks, self.prev_actions = get_blank_params(
            self.current_skill,
            self.device,
        )

        self.num_steps = 0
        self.num_transitions = 0
        self.next_skill_type = ""

    @property
    def current_skill_type(self):
        return self.current_skill["skill_type"]

    # Overload .to() method
    def to(self, device, *args):
        super().to(device, *args)
        for skill_name, skill_data in self.expert_skills.items():
            for k, v in skill_data.items():
                if k == "policy":
                    self.expert_skills[skill_name][k] = v.to(device)
        self.hidden_state = self.hidden_state.to(device)
        self.masks = self.masks.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.device = device

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        assert (
            config.NUM_PROCESSES == 1
        ), "SequentialExperts only works with 1 environment"
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            experts=config.SEQUENTIAL_EXPERTS,
        )

    def update_current_policy(self, next_skill_type):
        """Baton pass if observations reflect that current policy is done"""

        assert next_skill_type in self.expert_skills, (
            "SequentialExperts does not have the requested skill of "
            f"'{next_skill_type}'!"
        )

        self.current_skill = self.expert_skills[next_skill_type]

        self.hidden_state, self.masks, self.prev_actions = get_blank_params(
            self.current_skill,
            self.device,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False
    ):

        if masks[0] == 0.0:
            self.reset()
        elif self.next_skill_type != "":
            print(
                f"SequentialExperts changing from {self.current_skill_type}"
                f" to {self.next_skill_type}!"
            )
            self.update_current_policy(self.next_skill_type)
            self.next_skill_type = ""

        _, action, _, self.hidden_state = self.current_skill["policy"].act(
            observations,
            self.hidden_state,
            self.prev_actions,
            self.masks,
            deterministic=True,
        )
        self.prev_actions = action

        # We don't use these, but need to return them
        value, action_log_probs = None, None

        return value, action, action_log_probs, rnn_hidden_states
