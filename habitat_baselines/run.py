#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random

import numpy as np
import torch

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config, run_type)

    if run_type == "train":
        return trainer.train()
    elif run_type == "eval":
        return trainer.eval()

PATHS_TO_JUNK = {
    'LOG_FILE': '/private/home/naokiyokoyama/junk/train.log',
    'CHECKPOINT_FOLDER': '/private/home/naokiyokoyama/junk/',
    'TENSORBOARD_DIR': '/private/home/naokiyokoyama/junk/',
    'VIDEO_DIR': '/private/home/naokiyokoyama/junk/',
}

def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    # for idx, i in enumerate(opts):
    #     print(idx, i)
    if 'JUNK' in opts:
        idx = opts.index('JUNK')
        opts.pop(idx)
        opts.pop(idx)
        for k,v in PATHS_TO_JUNK.items():
            if k in opts:
                opts[opts.index(k)+1] = v
            else:
                opts.extend([k, v])

    # for idx, i in enumerate(opts):
    #     print(idx, i)
    if 'RL.POLICY.fuse_states' in opts:
        i = opts.index('RL.POLICY.fuse_states')
        opts[i+1] = opts[i+1].split(',')
        if len(opts[i+1]) == 1 and opts[i+1][0] == '':
            opts[i+1] = []

    config = get_config(exp_config, opts)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
