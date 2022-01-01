from collections import deque
from contextlib import ExitStack
import os
import os.path as osp
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ppo.sequential import get_blank_params

from habitat import logger

EXPERT_NAV_UUID = "expert_nav"
EXPERT_GAZE_UUID = "expert_gaze"
EXPERT_PLACE_UUID = "expert_place"
EXPERT_NULL_UUID = "undetermined"  # for when there is no applicable expert
EXPERT_UUIDS = [
    EXPERT_NAV_UUID,
    EXPERT_GAZE_UUID,
    EXPERT_PLACE_UUID,
    EXPERT_NULL_UUID,
]


@baseline_registry.register_trainer(name="bc")
class BehavioralCloningMoe(BaseRLTrainer):
    def __init__(self, config, *args, **kwargs):
        logger.add_filehandler(config.LOG_FILE)
        logger.info(f"Full config:\n{config}")

        self.config = config
        self.device = torch.device("cuda", 0)

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self.moe = None
        self.prev_actions = None
        self.masks = None
        self.num_actions = None
        self.envs = None
        self.success_deq = deque(maxlen=50)
        self.action_mse_deq = deque(maxlen=50)

        # Extract params from config
        self.batches_per_save = config.BATCHES_PER_CHECKPOINT
        self._batch_length = config.BATCH_LENGTH
        self.sl_lr = config.SL_LR
        self.policy_name = config.RL.POLICY.name
        self.total_num_steps = config.TOTAL_NUM_STEPS
        self.checkpoint_folder = config.CHECKPOINT_FOLDER
        self.tb_dir = config.TENSORBOARD_DIR
        self.bc_loss_type = config.BC_LOSS_TYPE
        self.load_weights = config.RL.DDPPO.pretrained

    def setup_teacher_student(self):
        # Envs MUST be instantiated first
        observation_space = self.envs.observation_spaces[0]
        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, obs_transforms
        )

        # MoE and its experts are loaded here
        policy_cls = baseline_registry.get_policy(self.policy_name)
        self.moe = policy_cls.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )

        # Load pretrained weights if provided
        if self.load_weights:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
            orig_state_dict = self.moe.state_dict()
            self.moe.load_state_dict(
                {
                    k: v if "expert" not in k else orig_state_dict[k]
                    for k, v in pretrained_state["state_dict"].items()
                }
            )

        self.moe.to(self.device)

        # Setup prev_actions, masks, and recurrent hidden states
        (
            self.rnn_hidden_states,
            self.masks,
            self.prev_actions,
        ) = get_blank_params(
            self.config, self.moe, self.device, num_envs=self.envs.num_envs
        )
        self.num_actions = self.prev_actions.shape[1]

    def get_model_params(self):
        return self.moe.model_params

    def get_action_and_loss(self, batch):
        # Extract teacher actions from the observations
        teacher_actions = []
        for idx, correct_skill_idx in enumerate(batch["correct_skill_idx"]):
            correct_action = (
                torch.ones(
                    self.num_actions, dtype=torch.float32, device=self.device
                )
                * 1e-6
            )
            correct_skill = EXPERT_UUIDS[int(correct_skill_idx)]
            if correct_skill == EXPERT_NULL_UUID:
                # Null action when an expert cannot be determined
                teacher_actions.append(correct_action)
                continue
            correct_action_arg = batch[correct_skill][idx]

            # For MoE_res, correct action is ZEROS for correct expert
            if correct_skill == EXPERT_NAV_UUID:
                correct_action[-2:] = correct_action_arg
            else:
                correct_action[: len(correct_action_arg)] = correct_action_arg
            teacher_actions.append(correct_action)
        teacher_actions = torch.cat(
            [t.reshape(1, self.num_actions) for t in teacher_actions], dim=0
        )

        # Get action loss and student actions
        if self.bc_loss_type == "log_prob":
            _, action_log_probs, _, _ = self.moe.evaluate_actions(
                batch,
                self.rnn_hidden_states,
                self.prev_actions,
                self.masks,
                teacher_actions,
            )
            with_stack = torch.no_grad
        else:
            with_stack = ExitStack

        with with_stack():
            _, actions, _, self.rnn_hidden_states = self.moe.act(
                batch,
                self.rnn_hidden_states,
                self.prev_actions,
                self.masks,
                deterministic=False,
            )
            self.prev_actions.copy_(actions)
            self.prev_actions = self.prev_actions.detach()
            self.rnn_hidden_states = self.rnn_hidden_states.detach()

        # Get action mse
        mse_loss = F.mse_loss(actions, teacher_actions)
        self.action_mse_deq.append(mse_loss.detach().cpu().item())

        if self.bc_loss_type == "log_prob":
            action_loss = -action_log_probs
        else:
            action_loss = mse_loss

        # Convert student actions into a dictionary for stepping envs
        step_actions = [
            self.moe.action_to_dict(act, index_env)
            for index_env, act in enumerate(actions.detach().cpu().unbind(0))
            # for index_env, act in enumerate(teacher_actions.unbind(0))
        ]

        return step_actions, action_loss

    def transform_observations(self, observations, masks):
        return self.moe.transform_obs(observations, masks)

    def train(self):
        # Andrew's code for VectorEnvs
        import sys

        sys.path.insert(0, "./")
        from orp_env_adapter import get_hab_envs, get_hab_args
        from method.orp_policy_adapter import HabPolicy

        policy = baseline_registry.get_policy(self.policy_name)
        if issubclass(policy, HabPolicy):
            policy = policy(self.config)
        else:
            policy = None
        self.envs, _ = get_hab_envs(
            self.config,
            "./config.yaml",
            False,  # is_eval
            spec_gpu=self.config.TORCH_GPU_ID,
            setup_policy=policy,
        )

        # Set up policies
        self.setup_teacher_student()

        # Set up optimizer
        optimizer = optim.Adam(self.get_model_params(), lr=self.sl_lr)

        # Set up tensorboard
        if self.tb_dir != "":
            print(f"Creating tensorboard at {self.tb_dir}...")
            os.makedirs(self.tb_dir, exist_ok=True)
            writer = SummaryWriter(self.tb_dir)
        else:
            writer = None

        # Start training
        observations = self.envs.reset()
        observations = self.transform_observations(observations, self.masks)
        batch = batch_obs(observations, device=self.device)

        batch_num = 0
        action_loss = 0
        iterations = int(self.total_num_steps // self.envs.num_envs)
        for iteration in range(1, iterations + 1):
            # Step environment using *student* actions
            step_actions, loss = self.get_action_and_loss(batch)
            outputs = self.envs.step(step_actions)

            # Format consequent observations for the next iteration
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            self.masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )
            observations = self.transform_observations(
                observations, self.masks
            )
            batch = batch_obs(observations, device=self.device)

            # Accumulate loss across batch
            action_loss += loss

            # Get episode stats
            for idx, done in enumerate(dones):
                if done:
                    self.success_deq.append(infos[idx]["ep_success"])

            if iteration % self._batch_length == 0:
                # Run backpropagation using accumulated loss across batch
                optimizer.zero_grad()
                action_loss = action_loss.mean() / float(self._batch_length)
                action_loss.backward()
                optimizer.step()

                # Print stats
                batch_num += 1
                mean_succ = (
                    0 if not self.success_deq else np.mean(self.success_deq)
                )
                logger.info(
                    f"iter: {iteration}\t"
                    f"batch_num: {batch_num}\t"
                    f"act_l: {action_loss.item():.4f}\t"
                    f"act_mse: {np.mean(self.action_mse_deq):.4f}\t"
                    f"mean_succ: {mean_succ:.4f}\t"
                )

                # Update tensorboard
                if writer is not None:
                    metrics_data = {"ep_success": mean_succ}
                    loss_data = {"action_loss": action_loss}
                    writer.add_scalars("metrics", metrics_data, batch_num)
                    writer.add_scalars("loss", loss_data, batch_num)

                # Reset loss
                action_loss = 0.0

                if batch_num % self.batches_per_save == 0:
                    # Save checkpoint
                    checkpoint = {
                        "state_dict": self.moe.state_dict(),
                        "config": self.config,
                        "iteration": iteration,
                        "batch": batch,
                    }
                    ckpt_id = int(batch_num / self.batches_per_save) - 1
                    filename = f"ckpt.{ckpt_id}_{batch_num}.pth"
                    ckpt_path = osp.join(self.checkpoint_folder, filename)
                    torch.save(checkpoint, ckpt_path)
                    print("Saved checkpoint:", ckpt_path)

        self.envs.close()
