#!/usr/bin/env python3,

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import os
import os.path as osp
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer, get_logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
)
from habitat_baselines.utils.env_utils import construct_envs

try:
    import rlf.rl.utils as rutils
    from rlf.exp_mgr.viz_utils import save_mp4
except:
    pass
import torch.nn as nn

from habitat.core.utils import try_cv2_import

cv2 = try_cv2_import()


@baseline_registry.register_trainer(name="ddppo")
@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPO
    actor_critic: Policy

    def __init__(self, config=None, run_type="train"):
        if run_type == "train":
            resume_state = load_resume_state(config)
            if resume_state is not None:
                config = resume_state["config"]

        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distirbuted if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(
        self, ppo_cfg: Config, del_envs=False
    ) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        keep_keys = {}

        ignore_keys = ppo_cfg.ignore_obs.split(",")
        observation_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if k not in ignore_keys
            }
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        print("Actor-critic architecture:\n", self.actor_critic)
        if hasattr(self.actor_critic, "fuse_states"):
            fuse_states_list = "\n - ".join(self.actor_critic.fuse_states)
            print("Fuse states:\n -", fuse_states_list)
        elif hasattr(self.actor_critic.net, "fuse_states"):
            fuse_states_list = "\n - ".join(self.actor_critic.net.fuse_states)
            print("Fuse states:\n -", fuse_states_list)
        self.obs_space = observation_space
        print("Observation space:")
        for k, v in observation_space.spaces.items():
            print(k, v.shape)
        if del_envs:
            self.envs.close()
            del self.envs
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
        if self.config.RL.DDPPO.pretrained:
            orig_state_dict = self.actor_critic.state_dict()
            try:
                self.actor_critic.load_state_dict(
                    {
                        k: v if "expert" not in k else orig_state_dict[k]
                        for k, v in pretrained_state["state_dict"].items()
                    }
                )
            except:
                try:
                    prefix = "actor_critic."
                    self.actor_critic.load_state_dict(
                        {
                            k[len(prefix) :]: v
                            if "expert" not in k
                            else orig_state_dict[k[len(prefix) :]]
                            for k, v in pretrained_state["state_dict"].items()
                        }
                    )
                except Exception as e:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!!!!!LOADING PRETRAINED WEIGHTS FAILED!!!!!!")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    raise e
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        if self.config.RL.POLICY.reset_moe_non_gating:
            print("Resetting MoE residual actor and critic...")
            from habitat_baselines.rl.ppo.moe_v2 import construct_mlp_base
            from habitat_baselines.utils.common import (
                GaussianNet,
                initialized_linear,
            )

            input_size = self.actor_critic.residual_actor[0].in_features
            hidden_size = self.actor_critic.residual_actor[0].out_features
            action_size = self.actor_critic.residual_actor[-1].mu.out_features
            self.actor_critic.residual_actor = nn.Sequential(
                *construct_mlp_base(input_size, hidden_size),
                GaussianNet(hidden_size, action_size),
            )
            self.actor_critic.critic = nn.Sequential(
                *construct_mlp_base(input_size, hidden_size),
                initialized_linear(hidden_size, 1, gain=np.sqrt(2)),
            )
            self.actor_critic.residual_actor.to(self.device)
            self.actor_critic.critic.to(self.device)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            use_second_optimizer=ppo_cfg.use_second_optimizer,
            second_optimizer_key=ppo_cfg.second_optimizer_key,
            lr_2=ppo_cfg.lr_2,
        )

    def _init_envs(self, is_eval, config=None):
        import sys

        sys.path.insert(0, "./")
        from method.orp_policy_adapter import HabPolicy
        from orp_env_adapter import get_hab_args, get_hab_envs

        if config is None:
            config = self.config

        if not self.is_simple_env():
            policy = baseline_registry.get_policy(config.RL.POLICY.name)
            if issubclass(policy, HabPolicy):
                policy = policy(config)
            else:
                policy = None
            self.envs, args = get_hab_envs(
                config,
                "./config.yaml",
                is_eval,
                spec_gpu=self.config.TORCH_GPU_ID,
                setup_policy=policy,
            )
        else:
            args = get_hab_args(
                config, "./config.yaml", spec_gpu=self.config.TORCH_GPU_ID
            )
            self.envs = construct_envs(
                config,
                get_env_class(self.config.ENV_NAME),
                workers_ignore_signals=is_slurm_batch_job(),
            )
        self.args = args

    def is_simple_env(self):
        return self.config.ENV_NAME != "Orp-v1"

    def _init_train(self):
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        # HACK: Memory error when envs are loaded before policy
        tmp_config = self.config.clone()
        tmp_config.defrost()
        tmp_config.NUM_PROCESSES = 1
        tmp_config.freeze()
        self._init_envs(is_eval=False, config=tmp_config)

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg, del_envs=True)

        # HACK: Memory error when envs are loaded before policy
        self._init_envs(is_eval=False)

        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1
        if hasattr(self.actor_critic, "policy_action_space"):
            self.policy_action_space = self.actor_critic.policy_action_space
        else:
            self.policy_action_space = self.envs.action_spaces[0]
        if hasattr(self.actor_critic, "net"):
            num_rnn_layers = self.actor_critic.net.num_recurrent_layers
        else:
            num_rnn_layers = 0
        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=num_rnn_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        if hasattr(self.actor_critic, "transform_obs"):
            observations = self.actor_critic.transform_obs(
                observations,
                masks=torch.zeros(
                    self.envs.num_envs, 1, dtype=torch.bool, device=self.device
                ),
                obs_transforms=self.obs_transforms,
            )
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        save_path = os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        print("Checkpointed to ", save_path)

        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                use_v = {k: dv for k, dv in v.items() if isinstance(k, str)}
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            use_v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    @profiling_wrapper.RangeContext("compute_and_step")
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if self.is_simple_env():
                self.envs.async_step_at(index_env, act.item())
            else:
                if hasattr(self.actor_critic, "action_to_dict"):
                    step_action = self.actor_critic.action_to_dict(
                        act, index_env, percent_done=self.percent_done()
                    )
                else:
                    step_action = {
                        "action": {
                            "action": act,
                            "percent_done": self.percent_done(),
                        },
                    }
                self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    @profiling_wrapper.RangeContext("_collect_environment_result")
    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]
        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        if hasattr(self.actor_critic, "transform_obs"):
            observations = self.actor_critic.transform_obs(
                observations,
                masks=not_done_masks,
                obs_transforms=self.obs_transforms,
            )

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        with profiling_wrapper.RangeContext("collect_env_to_tensors"):
            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
            for k, v_k in self._extract_scalars_from_infos(infos).items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                try:
                    self.running_episode_stats[k][env_slice] += v.where(
                        done_masks, v.new_zeros(())
                    )  # type: ignore
                except:
                    pass

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

            if self._static_encoder:
                with torch.no_grad():
                    batch["visual_features"] = self._encoder(batch)

            self.rollouts.insert(
                next_observations=batch,
                rewards=rewards,
                next_masks=not_done_masks,
                buffer_index=buffer_index,
            )

            self.rollouts.advance_rollout(buffer_index)

            self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        def get_k_count(k):
            scene_name = k.split("/")[0]
            count_k = scene_name + "_COUNT"
            if count_k in deltas:
                return max(deltas[count_k], 1.0)
            else:
                return deltas["count"]

        metrics = {
            k: v / get_k_count(k)
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            fps = self.num_steps_done / (
                (time.time() - self.t_start) + prev_time
            )
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(self.num_updates_done, fps)
            )
            writer.add_scalars(
                "metrics",
                {
                    "fps": fps,
                    "pth_time": self.pth_time,
                    "env_time": self.env_time,
                },
                self.num_steps_done,
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )
            self.pth_time = 0
            self.env_time = 0

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        _, _, n_tasks = get_distrib_size()
        self.config.defrost()
        self.config.N_TASKS = n_tasks
        self.config.freeze()

        with (
            get_logger(self.config, self.args, self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    print("Saving habitat resume state")
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        dist_entropy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                    )

                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            requeue_stats=requeue_stats,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        import sys

        sys.path.insert(0, "./")
        from method.orp_policy_adapter import HabPolicy
        from orp_env_adapter import ALL_SCENES

        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        if self.config.EVAL.EMPTY:
            ckpt_dict = {
                "state_dict": {
                    "actor_critic.dummy_param": nn.Parameter(
                        torch.tensor([0.0])
                    )
                }
            }
        else:
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        # Always keep the video directory the same.
        config.VIDEO_DIR = self.config.VIDEO_DIR
        if "EVAL_NODE" in config.TASK_CONFIG:
            config.TASK_CONFIG.EVAL_NODE = self.config.TASK_CONFIG.EVAL_NODE
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        ppo_cfg = config.RL.PPO

        use_video_option = self.config.VIDEO_OPTION[:]
        # if (
        #     False
        #     and (checkpoint_index + 1) % config.CHECKPOINT_RENDER_INTERVAL != 0
        # ):
        if len(use_video_option) == 0:
            use_video_option = []
            config.defrost()
            config.hab_high_render = False
            config.freeze()
        else:
            print("Rendering")
            config.defrost()
            config.hab_high_render = True
            config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if self.config.VIDEO_MAX_RENDER == 0:
            # A way to keep the high res camera but not save any videos
            use_video_option = []

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(True, config)
        self._setup_actor_critic_agent(ppo_cfg)

        try:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
        except:
            try:
                self.agent.actor_critic.load_state_dict(
                    ckpt_dict["state_dict"]
                )
            except Exception:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("WARNING: WEIGHTS WERE NOT PROPERLY LOADED!!")
                print("(this is usually OK for Sequential Experts)")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.actor_critic = self.agent.actor_critic

        if self.actor_critic is not None and isinstance(
            self.agent.actor_critic, HabPolicy
        ):
            # For custom policy.
            self.agent.actor_critic.init(
                self.envs.observation_spaces[0],
                self.envs.action_spaces[0],
                self.args,
            )
            self.agent.actor_critic.set_env_ref(self.envs)

        ## For debugging a particular episode index.
        # print('Searching for particular episode')
        ##while True:
        # for _ in range(13):
        #    observations = self.envs.reset()
        #    print('Episode', self.envs.current_episodes()[0].episode_id)
        #    #if self.envs.current_episodes()[0].episode_id == '22':
        #    #    break
        # print('Found matching environment')
        ## IF USING THE ABOVE, COMMENT OUT THE BELOW.

        observations = self.envs.reset()
        if hasattr(self.actor_critic, "transform_obs"):
            observations = self.actor_critic.transform_obs(
                observations,
                masks=torch.zeros(
                    self.envs.num_envs, 1, dtype=torch.bool, device=self.device
                ),
                obs_transforms=self.obs_transforms,
            )
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        if hasattr(self.actor_critic, "net"):
            num_rnn_layers = self.actor_critic.net.num_recurrent_layers
        else:
            num_rnn_layers = 0

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            num_rnn_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        if self.is_simple_env():
            ac_shape = 1
            ac_dtype = torch.float32
        else:
            if hasattr(self.actor_critic, "policy_action_space"):
                ac_shape = self.actor_critic.policy_action_space.shape[0]
                ac_dtype = torch.float32
            else:
                if hasattr(self.envs.action_spaces[0], "spaces"):
                    ac_shape = len(self.envs.action_spaces[0].spaces)
                    ac_dtype = torch.long
                else:
                    ac_shape = self.envs.action_spaces[0].shape[0]
                    ac_dtype = torch.float32

        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            ac_shape,
            device=self.device,
            dtype=ac_dtype,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        stats_counts = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]
        if (
            isinstance(self.actor_critic, HabPolicy)
            and self.config.PREFIX != "debug"
        ):
            # Not debug because it is annoying when a new folder is
            # created every time for the videos while debugging.
            step_id = (
                self.actor_critic.mod_policy.get_total_num_training_steps()
            )

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()

        use_video_dir = os.path.join(
            self.config.VIDEO_DIR,
            "ckpt_%i_%i" % (checkpoint_index, int(step_id)),
        )
        if len(use_video_option) > 0 and not os.path.exists(use_video_dir):
            os.makedirs(use_video_dir)

        should_save_replay = False
        if self.config.NUM_PROCESSES == 1:
            single_sim = rutils.get_env_attr(self.envs.env.envs[0], "sim")
            should_save_replay = single_sim.habitat_config.HABITAT_SIM_V0.get(
                "ENABLE_GFX_REPLAY_SAVE", False
            )

        cur_render = 0
        num_ended = 0
        num_successful = 0
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    # deterministic=True,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            if self.is_simple_env():
                step_data = [a.item() for a in actions.to(device="cpu")]
            else:
                if hasattr(self.actor_critic, "action_to_dict"):
                    step_data = [
                        self.actor_critic.action_to_dict(act, index_env)
                        for index_env, act in enumerate(
                            actions.cpu().unbind(0)
                        )
                    ]
                elif self.config.RL.POLICY.name == "NavGazeMixtureOfExperts":
                    step_data = self.actor_critic.choose_mix_of_actions(
                        actions
                    )
                elif self.config.RL.POLICY.name == "SequentialExperts":
                    step_data = [
                        {
                            "action": a.numpy(),
                            "skill_type": self.actor_critic.current_skill_type,
                        }
                        for a in actions.to(device="cpu")
                    ]
                else:
                    step_data = [
                        {"action": a.numpy()} for a in actions.to(device="cpu")
                    ]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            if hasattr(self.actor_critic, "transform_obs"):
                observations = self.actor_critic.transform_obs(
                    observations,
                    masks=not_done_masks,
                    obs_transforms=self.obs_transforms,
                )

            if self.config.RL.POLICY.name == "SequentialExperts":
                self.actor_critic.next_skill_type = infos[0]["next_skill_type"]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            if len(use_video_option) > 0:
                if self.is_simple_env():
                    frames = [
                        observations_to_image(observations[i], infos[i])
                        for i in range(len(infos))
                    ]
                else:
                    frames = self.envs.render(mode="rgb_array")
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # WE WANT TO RENDER THE FINAL FRAME. But only for manip tasks
                # since we display statistics at the end.
                if len(use_video_option) > 0:
                    frame = frames[i]
                    if self.is_simple_env():
                        if not_done_masks[i].item() != 0:
                            rgb_frames[i].append(np.flip(frame, 0))
                    else:
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = dict()
                    fsm_dat = {}
                    if hasattr(self.actor_critic, "mod_policy"):
                        # Stats from the modular policy such as failed modules.
                        fsm_dat = self.actor_critic.mod_policy.get_skill_data()
                        if infos[i]["ep_success"] == 1.0:
                            # Nothing could have been a failure
                            fsm_dat = {
                                k: 0.0 if "failure" in k else v
                                for k, v in fsm_dat.items()
                            }
                        episode_stats.update(fsm_dat)
                    episode_stats["reward"] = current_episode_reward[i].item()
                    extracted = self._extract_scalars_from_info(infos[i])
                    episode_stats.update(extracted)
                    for k in {**extracted, **fsm_dat}:
                        stats_counts[k] += 1
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    num_ended += 1
                    if episode_stats["ep_success"] == 1:
                        num_successful += 1
                    if len(use_video_option) > 0:
                        name_conversion = {
                            "ep_success": "succ",
                            "ep_constraint_violate": "const_violate",
                            "spl": "spl",
                            "ep_accum_force_end": "force_end",
                            "node_idx": "node",
                            # MP failure highest-level bins
                            "plan_failure": "plan_fail",
                            "plan_guess": "plan_guess",
                            "execute_failure": "ex_fail",
                            "nav1": "nav1",
                            "nav2": "nav2",
                            "gaze": "gaze",
                        }
                        # filename
                        fname_metrics = {
                            name_conversion[k]: v
                            for k, v in episode_stats.items()
                            if k in list(name_conversion.keys())
                        }
                        fname_metrics["reward"] = episode_stats["reward"]
                        if "scene_name" in infos[i]:
                            scene_name = ALL_SCENES[infos[i]["scene_name"]]
                            fname_metrics["name"] = scene_name

                        video_name = generate_video(
                            video_option=use_video_option,
                            video_dir=use_video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=fname_metrics,
                            tb_writer=writer,
                            fps=self.config.VIDEO_FPS,
                        )

                        if len(rgb_frames[i]) <= 3:
                            print("SUPER SHORT!")
                            # exit()

                        rgb_frames[i] = []
                        if (
                            self.config.VIDEO_MAX_RENDER > 0
                            and cur_render > self.config.VIDEO_MAX_RENDER
                        ):
                            # Turn off rendering.
                            self.config.defrost()
                            use_video_option = []
                            self.config.freeze()
                        cur_render += 1
                        replay_dir = use_video_dir
                        if should_save_replay:
                            single_sim._sim.gfx_replay_manager.write_saved_keyframes_to_file(
                                video_name + ".json"
                            )
                    else:
                        print(
                            "Stats:\tep_id:",
                            current_episodes[i].episode_id,
                            "success:",
                            episode_stats["ep_success"],
                            "num_steps:",
                            episode_stats["num_steps"],
                        )
                    print(
                        f"Success rate: {num_successful/num_ended*100:.2f}% "
                        f"({num_successful} out of {num_ended})"
                    )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        should_save_std = isinstance(self.agent.actor_critic, HabPolicy)
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key in ["reward", "count"]:
                use_count = num_episodes
            else:
                if stat_key not in stats_counts:
                    raise ValueError(f"{stat_key} not present in count dict")
                use_count = stats_counts[stat_key]

            values = np.array(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )
            mean_val = sum(values) / use_count
            aggregated_stats[stat_key] = mean_val

            if self.config.EVAL_SAVE_STD:
                if self.config.hab_multi_scene:
                    raise ValueError(
                        "Cannot compute STD with multi-scene breakdown"
                    )
                if "profile" in stat_key:
                    continue
                std = np.sqrt(np.mean(np.abs(values - mean_val) ** 2))
                aggregated_stats[stat_key + "_std"] = std

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
        if "HabFormatWrapper" in str(type(self.envs)):
            sim = rutils.get_env_attr(self.envs.env.envs[0], "_sim")
            sim.close(destroy=True)
        del self.envs
        stats_episodes_json = {}
        for k, v in stats_episodes.items():
            stats_episodes_json[k[1]] = v
        ckpt_str = osp.basename(checkpoint_path)[:-4].replace(".", "_")
        if "JSON_PATH" not in self.config:
            stats_json_path = self.config.LOG_FILE[:-4] + f"_{ckpt_str}.json"
        else:
            stats_json_path = self.config.JSON_PATH

        print("Saving json to:", stats_json_path)
        with open(stats_json_path, "w") as f:
            json.dump(stats_episodes_json, f, indent=4, sort_keys=True)
