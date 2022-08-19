import torch
from gym import spaces
from gym.spaces import Box

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import PointNavBaselinePolicy

ARM_ACTIONS = 4  # 4 controllable joints
BASE_ACTIONS = 2  # linear and angular vel
EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"
EXPERT_PLACE_UUID = "expert_place"
EXPERT_MASKS_UUID = "expert_masks"
VISUAL_FEATURES_UUID = "visual_features"
EXPERT_ACTIONS_UUID = "expert_actions"
CORRECTIONS_UUID = "corrections"

"""
This policy has 6 or 7 actions, depending on whether a gating action is used.
During evaluate_actions, we cannot use the log probs of the actions that were
not selected for execution.
During action_to_dict, the gating action must take effect.
"""


@baseline_registry.register_policy
class RelmogenPolicy(PointNavBaselinePolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        use_gating: bool = True,
        **kwargs,
    ):
        # For RolloutStorage in ppo_trainer.py
        self.policy_action_space = action_space
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            **kwargs,
        )
        self.use_gating = use_gating

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        goal_hidden_size = config.RL.PPO.get("goal_hidden_size", 0)
        num_actions = 7 if config.RL.POLICY.use_gating else 6
        actual_action_space = Box(-1.0, 1.0, (num_actions,))
        return cls(
            observation_space=observation_space,
            action_space=actual_action_space,
            hidden_size=config.RL.PPO.hidden_size,
            goal_hidden_size=goal_hidden_size,
            fuse_states=config.RL.POLICY.fuse_states,
            force_blind=config.RL.POLICY.force_blind,
            init=config.RL.POLICY.get("init", True),
            use_gating=config.RL.POLICY.use_gating,
        )

    def action_to_dict(self, action, *args, **kwargs):
        if self.use_gating:
            step_action = action.to(torch.device("cpu"))[:-1]
            gate = action[-1]
            if gate > 0:
                step_action[ARM_ACTIONS:] = 0.0
            else:
                step_action[:ARM_ACTIONS] = 0.0
        else:
            step_action = action.to(torch.device("cpu"))
            assert step_action.shape[0] == ARM_ACTIONS + BASE_ACTIONS
            gate = 0.0

        # Add expert actions as action_args for reward calculation in RLEnv
        step_data = {
            "action": {
                "action": step_action,
                "gate": gate,
                **kwargs,
            }
        }

        return step_data

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        get_sum=False,
    ):
        assert not get_sum
        (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
        ) = super().evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            get_sum=get_sum,
        )
        if self.use_gating:
            gates = action[:, -1].unsqueeze(1)

            num_envs = gates.shape[0]
            gates_repeated = gates.repeat(1, ARM_ACTIONS + BASE_ACTIONS)
            # We ALWAYS want the log_prob of the gate action (hence .ones())
            gates_repeated = torch.cat(
                [
                    gates_repeated,
                    torch.ones(num_envs, 1, device=gates.device),
                ],
                dim=1,
            )

            action_log_probs = torch.where(
                torch.gt(gates_repeated, 0),
                action_log_probs,
                torch.zeros_like(action_log_probs),
            )
        assert action_log_probs.ndim > 1 and distribution_entropy.ndim > 1
        return value, action_log_probs, distribution_entropy, rnn_hidden_states
