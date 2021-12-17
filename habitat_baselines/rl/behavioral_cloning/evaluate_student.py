import os
import glob
import time
import torch
import tqdm
import wandb
import argparse
import yaml 
import socket
import numpy as np

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.behavioral_cloning.agents import (
    D4Teacher,
    GaussianStudent,
)
from habitat_baselines.rl.behavioral_cloning.behavioral_cloning_v5 import (
    continuous_to_step_actions
)
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.utils import (
    batch_obs
)

from habitat import Config, logger

import habitat

# import habitat.tasks.nav.cont_ctrl

@baseline_registry.register_trainer(name="eval_student")
class EvalStudent(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config, ckpt_path):

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()
        
        self.config = config
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))

        self.student = GaussianStudent(config.RL)
        
        # self.student.actor_critic.load_state_dict(
        #     {
        #         k[len("actor_critic.") :]: v
        #         for k, v in torch.load(ckpt_path)["state_dict"].items()
        #     }
        # )
        self.student.actor_critic.load_state_dict(
            torch.load(ckpt_path)["state_dict"]
        )
        config.defrost()
        # Use continuous action space
        config.TASK_CONFIG.TASK.ACTIONS.CONT_MOVE = habitat.config.Config()
        config.TASK_CONFIG.TASK.ACTIONS.CONT_MOVE.TYPE = "ContMove"
        config.TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG = "ContCtrlSpace"
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[1] = 'CONT_MOVE'

        config.TASK_CONFIG.DATASET.DATA_PATH = '/nethome/nyokoyama3/n/datasets/pointnav_gibson/v1_splitup/{split}/{split}.json.gz'
        config.NUM_PROCESSES = 140
        # config.NUM_PROCESSES = 14
        config.freeze()
        logger.info(f"env config: {config}")

        self.ckpt_path = ckpt_path

    def run(self):
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.student.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # Student previous actions
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )

        number_of_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_episodes == -1:
            number_of_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_episodes:
                logger.warn(
                    f"Config specified {number_of_episodes} episodes"
                    f", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_episodes)
        self.student.actor_critic.eval()
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        all_spl = []
        while (
            len(stats_episodes) < number_of_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.student.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

            prev_actions.copy_(actions)

            step_actions = continuous_to_step_actions(actions)

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

            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    all_spl.append(infos[i]['spl'])
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
                torch.zeros(self.config.NUM_PROCESSES), # current_episode_reward
                prev_actions,
                batch,
                rgb_frames=[[] for _ in range(self.config.NUM_PROCESSES)],
            )
        
        all_succ = [1. if i > 0. else 0. for i in all_spl]
        avg_spl = np.mean(all_spl)
        avg_succ = np.mean(all_succ)
        wandb_data = {
            'avg_succ': avg_succ,
            'avg_spl': avg_spl,
        }
        wandb_step = int(os.path.basename(self.ckpt_path)[:-len('.pth')])
        wandb.log(wandb_data, step=wandb_step*76)
        print(wandb_data)

        self.envs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='run_name')
    parser.add_argument('config_file', help='config yaml file')
    args = parser.parse_args()

    config = get_config(args.config_file)
    exp_name = os.path.basename(args.config_file)[:-len('.yaml')]

    with open(args.config_file) as f:
        bc_config = yaml.load(f)['BC']

    bc_config['skynet_node'] = socket.gethostname()

    evaluated_ckpts = set()

    wandb.login()
    with wandb.init(
        project='irl_project_v3_eval',
        config=bc_config,
        # mode="disabled"
    ):
        wandb.run.name = args.run_name
        all_ckpts = glob.glob(os.path.join(
            config.CHECKPOINT_FOLDER,
            '*.pth'
        ))
        all_ckpts = sorted(
            all_ckpts,
            key=lambda x: int(os.path.basename(x)[:-len('.pth')])
        )
        for ckpt_path in all_ckpts:
            print(f'\n\n\nEVALUATING {ckpt_path}\n\n\n')
            evaluated_ckpts.add(ckpt_path)
            es = EvalStudent(config, ckpt_path)
            es.run()
        # print(f'Monitoring {config.CHECKPOINT_FOLDER}')
        # while True:
        #     all_ckpts = glob.glob(os.path.join(
        #         config.CHECKPOINT_FOLDER,
        #         '*.pth'
        #     ))
        #     all_ckpts = list(filter(
        #         lambda x: x not in evaluated_ckpts,
        #         all_ckpts
        #     ))
        #     for ckpt_path in sorted(all_ckpts):
        #         print(f'\n\n\nEVALUATING {ckpt_path}\n\n\n')
        #         evaluated_ckpts.add(ckpt_path)
        #         es = EvalStudent(config, ckpt_path)
        #         es.run()
        #         del es
        #     time.sleep(10)