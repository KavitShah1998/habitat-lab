import os
import argparse
import tqdm
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict, deque

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.behavioral_cloning.agents import (
    D4Teacher,
    GaussianStudent
)
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.utils import batch_obs

from habitat import Config, logger

BATCH_LENGTH = 4
LAST_TEACHER_BATCH = 20000

def discrete_to_continuous(actions, device):
    continuous_actions = []
    for a in actions:
        if a[0].item() == 0: # STOP
            continuous_actions.append([1.,0.])
        elif a[0].item() == 1: # FORWARD
            continuous_actions.append([-1.,0.])
        elif a[0].item() == 2: # LEFT
            continuous_actions.append([1.,1.])
        elif a[0].item() == 3: # RIGHT
            continuous_actions.append([1.,-1.])
    return torch.tensor(continuous_actions, device=device)
def continuous_to_discrete(actions, device):
    discrete_actions = []
    for a in actions.detach().clone():
        a_tanh = torch.tanh(a)
        lin_vel, ang_vel = a_tanh[0].item(), a_tanh[1].item()
        d_action = min(
            [(0,1.,0.), (1,-1.,0.), (2,1.,1.), (3,1.,-1.)],
            key=lambda x:(lin_vel-x[1])**2+(ang_vel-x[2])**2
        )[0]
        discrete_actions.append([d_action])
    return torch.tensor(discrete_actions, device=device, dtype=torch.long)
def copy_batch(batch, device):
    batch_copy = defaultdict(list)
    for sensor in batch:
        batch_copy[sensor] = batch[sensor].detach().clone()
    return batch_copy
def cat_batch(batch_list, device):
    batch = defaultdict(list)

    for b in batch_list:
        for sensor in b:
            batch[sensor].append(b[sensor])

    for sensor in batch:
        batch[sensor] = torch.cat(batch[sensor], dim=0)
    return batch

@baseline_registry.register_trainer(name="behavioral_cloning")
class BehavioralCloning(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        logger.info(f"env config: {config}")

        # Faster loading
        # config.defrost()
        # config.TASK_CONFIG.DATASET.SPLIT = 'val'
        # config.freeze()

        self.config = config
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        self.teacher = D4Teacher(config.RL)
        self.student = GaussianStudent(config.RL)

        # self.student.actor_critic.load_state_dict(
        #      torch.load(CHECKPOINT_PATH)['state_dict']
        # )

    def train(self):
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        # Teacher tensors
        teacher_hidden_states = torch.zeros(
            self.teacher.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        teacher_prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        # Student tensors
        student_hidden_states = torch.zeros(
            self.student.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        student_prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 2, device=self.device
        )

        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        self.teacher.actor_critic.eval()
        
        # self.optimizer = optim.Adam(
        #     list(filter(
        #         lambda p: p.requires_grad,
        #         self.student.actor_critic.parameters()
        #     )),
        #     lr=2.5e-4,
        #     eps=1e-5,
        # )
        self.optimizer = optim.SGD(
            list(filter(
                lambda p: p.requires_grad,
                self.student.actor_critic.parameters()
            )),
            lr=2.5,
        )
        batch_num = 0
        spl_deq = deque([0.], maxlen=100)
        all_done = 0
        action_loss = 0
        for iteration in range(1,int(1e8)):
            current_episodes = self.envs.current_episodes()

            in_batch = copy_batch(batch, device=self.device)
            in_hidden = student_hidden_states.detach().clone()
            in_prev_actions = student_prev_actions.detach().clone()
            in_not_done = not_done_masks.detach().clone()

            with torch.no_grad():
                (
                    _,
                    teacher_actions,
                    _,
                    teacher_hidden_states,
                ) = self.teacher.actor_critic.act(
                    batch,
                    teacher_hidden_states,
                    teacher_prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

            (
                _,
                student_actions,
                _,
                student_hidden_states,
            ) = self.student.actor_critic.act(
                in_batch,
                in_hidden,
                in_prev_actions,
                in_not_done,
                deterministic=True,
            )

            # Loss and update
            student_actions_tanh = torch.tanh(student_actions)
            teacher_labels = discrete_to_continuous(
                teacher_actions,
                device=self.device
            )
            action_loss += (student_actions_tanh - teacher_labels).pow(2).mean()

            if iteration % BATCH_LENGTH == 0:
                self.optimizer.zero_grad()
                action_loss /= float(BATCH_LENGTH)
                action_loss.backward()
                self.optimizer.step()
                batch_num += 1
                print(
                    f'#{batch_num}'
                    f'  action: {action_loss.item():.4f}'
                    f'  avg_spl: {np.mean(spl_deq):.4f}'
                )

                # if batch_num % 50 == 0:
                #     checkpoint = {
                #         "state_dict": self.student.actor_critic.state_dict(),
                #         "config": self.config,
                #     }

                #     file_name = (
                #         f'{batch_num:03d}_'
                #         f'{action_loss.item():.4f}_'
                #         f'{np.mean(spl_deq):.4f}'
                #         '.ckpt'
                #     )
                #     if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
                #         os.makedirs(self.config.CHECKPOINT_FOLDER)
                #     torch.save(
                #         checkpoint,
                #         os.path.join(
                #             self.config.CHECKPOINT_FOLDER,
                #             file_name
                #         )
                #     )
                action_loss = 0

            # Step environment
            teacher_thresh = 1.0 - float(batch_num)/float(LAST_TEACHER_BATCH)
            teacher_drives = np.random.rand() < teacher_thresh
            if teacher_drives:
                teacher_prev_actions.copy_(teacher_actions)
                student_prev_actions.copy_(
                    discrete_to_continuous(
                        teacher_actions,
                        device=self.device
                    )
                )
                step_actions = [a[0].item() for a in teacher_actions]
            else:
                student_prev_actions.copy_(student_actions)
                teacher_actions_converted = continuous_to_discrete(
                    student_actions,
                    device=self.device
                )
                teacher_prev_actions.copy_(teacher_actions_converted)
                step_actions = [a[0].item() for a in teacher_actions_converted]

            outputs = self.envs.step(step_actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            for idx, done in enumerate(dones):
                if done:
                    spl_deq.append(infos[idx]['spl'])

            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

        self.envs.close()

if __name__ == '__main__':
    config = get_config(
        'habitat_baselines/config/pointnav/behavioral_cloning.yaml'
    )
    d = BehavioralCloning(config)
    d.train()