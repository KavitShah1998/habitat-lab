PROJECT_NAME = 'irl_critic_loss'

import os
import argparse
import tqdm
import numpy as np
import yaml
import socket
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
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

import habitat
from habitat import Config, logger

torch.backends.cudnn.enabled = False

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
    for a in actions:
        lin_vel, ang_vel = float(a[0].item()), float(a[1].item())
        lin_vel, ang_vel = np.tanh(lin_vel), np.tanh(ang_vel)
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

def discrete_to_step_actions(actions):
    step_actions = []
    for a in actions:
        a_item = int(a.item())
        lin_vel, ang_vel = 0., 0.
        if a_item == 1:
            lin_vel = -0.25
        elif a_item == 2:
            ang_vel = np.pi/180*10
        elif a_item == 3:
            ang_vel = -np.pi/180*10
        step_action = lin_ang_to_step_action(lin_vel, ang_vel)
        step_actions.append(step_action)
    return step_actions

def continuous_to_step_actions(actions):
    step_actions = []
    for a in actions:
        lin_vel, ang_vel = np.tanh(a[0].item()), np.tanh(a[1].item())
        lin_vel = (lin_vel-1.)/2.*0.25
        ang_vel *= np.pi/180*10
        step_action = lin_ang_to_step_action(lin_vel, ang_vel)
        step_actions.append(step_action)
    return step_actions

def lin_ang_to_step_action(lin_vel, ang_vel):
    return {
        'action': { 
            'action': 'CONT_MOVE',
            'action_args': {
                'linear_velocity': lin_vel,
                'angular_velocity': ang_vel,
                'time_step' : 1.0,
                'allow_sliding': True,
            }
        }
    }

@baseline_registry.register_trainer(name="behavioral_cloning")
class BehavioralCloning(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config, debug=False):
        torch.backends.cudnn.enabled = False
        logger.info(f"env config: {config}")

        self.config = config
        self.device = torch.device("cuda", 0)
        self.teacher = D4Teacher(config.RL)
        self.teacher.actor_critic.eval()
        self.student = GaussianStudent(config.RL)

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        config.defrost()
        # Use continuous action space
        config.TASK_CONFIG.TASK.ACTIONS.CONT_MOVE = habitat.config.Config()
        config.TASK_CONFIG.TASK.ACTIONS.CONT_MOVE.TYPE = "ContMove"
        config.TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG = "ContCtrlSpace"
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[1] = 'CONT_MOVE'

        if debug:
            config.BC.NUM_PROCESSES = 4
            config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT

        config.NUM_PROCESSES = config.BC.NUM_PROCESSES
        config.freeze()

        self._num_processes = config.BC.NUM_PROCESSES
        self._deque_length = config.BC.DEQUE_LENGTH
        self._batch_length = config.BC.BATCH_LENGTH

    def train(self):
        torch.backends.cudnn.enabled = False
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        not_done_masks = torch.zeros(
            self._num_processes, 1, device=self.device
        )

        # Teacher tensors
        teacher_hidden_states = torch.zeros(
            self.teacher.actor_critic.net.num_recurrent_layers,
            self._num_processes,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        teacher_prev_actions = torch.zeros(
            self._num_processes, 1, device=self.device, dtype=torch.long
        )
        # Student tensors
        student_hidden_states = torch.zeros(
            self.student.actor_critic.net.num_recurrent_layers,
            self._num_processes,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )
        student_prev_actions = torch.zeros(
            self._num_processes, 2, device=self.device
        )
        
        opt = optim.SGD if self.config.BC.SGD else optim.Adam

        self.optimizer = opt(
            list(filter(
                lambda p: p.requires_grad,
                self.student.actor_critic.parameters()
            )),
            lr=self.config.BC.SL_LR,
        )
        batch_num = 0
        spl_deq = deque(maxlen=self._deque_length)
        action_loss = 0
        max_spl = 0
        save_at_iteration = self.config.BC.SAVE_ITERATIONS
        for iteration in range(
            1,
            self.config.BC.NUM_ITERATIONS//self._num_processes+1
        ):
            current_episodes = self.envs.current_episodes()

            # in_batch = copy_batch(batch, device=self.device)
            in_hidden = student_hidden_states.detach().clone()
            in_prev_actions = student_prev_actions.detach().clone()
            in_not_done = not_done_masks.detach().clone()

            with torch.no_grad():
                (
                    teacher_value,
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
                student_value,
                student_actions,
                _,
                student_hidden_states,
            ) = self.student.actor_critic.act(
                batch,
                in_hidden,
                in_prev_actions,
                in_not_done,
                deterministic=True,
            )

            # Loss and update
            value_loss += F.mse_loss(
                teacher_value, 
                student_value,
                reduce='mean'
            )

            student_actions_tanh = torch.tanh(student_actions)
            teacher_labels = discrete_to_continuous(
                teacher_actions,
                device=self.device
            )
            action_loss += F.mse_loss(
                student_actions_tanh, 
                teacher_labels,
                reduce='mean'
            )

            if self.config.BC.LAST_TEACHER_BATCH == -1:
                teacher_thresh = 1.0
            else:
                teacher_thresh = (
                    1.0 -
                    iteration*self._num_processes
                    / self.config.BC.LAST_TEACHER_BATCH
                )
            if iteration % self._batch_length == 0:
                self.optimizer.zero_grad()
                action_loss /= float(self._batch_length)
                action_loss.backward()
                self.optimizer.step()
                batch_num += 1
                mean_spl = np.mean(spl_deq) if spl_deq else 0
                mean_succ = np.mean(
                    [1. if i>0 else 0. for i in spl_deq]
                ) if spl_deq else 0
                print(
                    f'#{batch_num}'
                    f'  action: {action_loss.item():.4f}'
                    f'  avg_spl: {mean_spl:.4f}'
                    f'  avg_succ: {mean_succ:.4f}'
                    f'  teacher_thresh: {max(0,teacher_thresh):.4f}'
                )

                wandb_data = {
                    'action_loss':action_loss.item(),
                    'avg_spl': mean_spl,
                    'teacher_thresh': max(0,teacher_thresh),
                }
                wandb.log(wandb_data, step=iteration*self._num_processes)

            if iteration*self._num_processes > save_at_iteration:
                save_at_iteration += self.config.BC.SAVE_ITERATIONS
                checkpoint = {
                    "state_dict": self.student.actor_critic.state_dict(),
                    "config": self.config,
                }

                file_name = (
                    f'{int(iteration)}'
                    '.pth'
                )
                ckpt_path = os.path.join(
                    self.config.CHECKPOINT_FOLDER,
                    file_name
                )
                torch.save(
                    checkpoint,
                    ckpt_path
                )
            action_loss = 0

            # Step environment
            teacher_drives = np.random.rand() < teacher_thresh
            if teacher_drives:
                teacher_prev_actions.copy_(teacher_actions)
                student_prev_actions.copy_(
                    discrete_to_continuous(
                        teacher_actions,
                        device=self.device
                    )
                )
                step_actions = discrete_to_step_actions(teacher_actions)
            else:
                teacher_actions_converted = continuous_to_discrete(
                    student_actions,
                    # device=self.device
                )
                teacher_prev_actions.copy_(teacher_actions_converted)
                student_prev_actions.copy_(student_actions.detach().clone())
                step_actions = continuous_to_step_actions(student_actions)

            outputs = self.envs.step(step_actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # Update moving average of spl
            for idx, done in enumerate(dones):
                if done:
                    spl_deq.append(infos[idx]['spl'])
                    teacher_hidden_states[:,idx,:] = torch.zeros(
                        self.teacher.actor_critic.net.num_recurrent_layers,
                        self.config.RL.PPO.hidden_size,
                        device=self.device,
                    )
                    student_hidden_states[:,idx,:] = torch.zeros(
                        self.student.actor_critic.net.num_recurrent_layers,
                        self.config.RL.PPO.hidden_size,
                        device=self.device,
                    )
                    student_prev_actions[idx] = torch.zeros(
                        2,
                        device=self.device,
                    )
                    teacher_prev_actions[idx] = torch.zeros(
                        1,
                        device=self.device,
                        dtype=torch.long
                    )

        self.envs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='run_name')
    parser.add_argument('config_file', help='config yaml file')
    parser.add_argument('-d','--debug' action='store_true')
    args = parser.parse_args()

    config = get_config(args.config_file)
    exp_name = os.path.basename(args.config_file)[:-len('.yaml')]
    bc = BehavioralCloning(config, debug=args.debug)

    with open(args.config_file) as f:
        bc_config = yaml.load(f)['BC']

    bc_config['skynet_node'] = socket.gethostname()

    wandb.login()

    if args.debug:
        with wandb.init(
            project=PROJECT_NAME,
            mode='disabled',
            config=bc_config
        ):
            wandb.run.name = args.run_name
            bc.train()
    else:
        with wandb.init(
            project=PROJECT_NAME,
            config=bc_config
        ):
            wandb.run.name = args.run_name
            bc.train()
