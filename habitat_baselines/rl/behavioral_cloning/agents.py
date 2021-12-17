import torch
import numpy as np
import os
from gym.spaces import Dict, Box, Discrete

from habitat import Config
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy,
)

class D4Teacher:
    def __init__(self, rl_cfg: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device(
            "cuda",
            int(os.environ["SLURM_LOCALID"])
        )

        depth_256_space = Dict({
            'depth': Box(low=0., high=1., shape=(256,256,1)),
            'pointgoal_with_gps_compass': Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ) 
        })
        self.actor_critic = PointNavResNetPolicy(
            observation_space=depth_256_space,
            action_space=Discrete(4),
            hidden_size=rl_cfg.PPO.hidden_size,
            rnn_type=rl_cfg.DDPPO.rnn_type,
            num_recurrent_layers=rl_cfg.DDPPO.num_recurrent_layers,
            backbone='resnet50',
            normalize_visual_inputs=False,
            dim_actions=4,
            action_distribution='categorical',
        )
        self.actor_critic.to(self.device)

        pretrained_state = torch.load(
            rl_cfg.DDPPO.pretrained_weights, map_location="cpu"
        )
        self.actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in pretrained_state["state_dict"].items()
            }
        )

class GaussianStudent:
    def __init__(self, rl_cfg: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device(
            "cuda",
            int(os.environ["SLURM_LOCALID"])
        )

        depth_256_space = Dict({
            'depth': Box(low=0., high=1., shape=(256,256,1)),
            'pointgoal_with_gps_compass': Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ) 
        })
        action_space = Box(
            np.array([float('-inf'), float('-inf')]),
            np.array([float('inf'),  float('inf')])
        )
        self.actor_critic = PointNavResNetPolicy(
            observation_space=depth_256_space,
            action_space=action_space,
            hidden_size=rl_cfg.PPO.hidden_size,
            rnn_type=rl_cfg.DDPPO.rnn_type,
            num_recurrent_layers=rl_cfg.DDPPO.num_recurrent_layers,
            backbone='resnet50',
            normalize_visual_inputs=False,
            dim_actions=2,
            action_distribution='gaussian',
        )
        self.actor_critic.to(self.device)

        pretrained_state = torch.load(
            rl_cfg.DDPPO.pretrained_weights, map_location="cpu"
        )
        prefix = "actor_critic.net.visual_encoder."
        self.actor_critic.net.visual_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        for param in self.actor_critic.net.visual_encoder.parameters():
            param.requires_grad_(False)

if __name__ == '__main__':
    from habitat_baselines.config.default import get_config
    config = get_config(
        'habitat_baselines/config/pointnav/behavioral_cloning.yaml'
    )
    d = D4Teacher(config.RL)
    g = GaussianStudent(config.RL)