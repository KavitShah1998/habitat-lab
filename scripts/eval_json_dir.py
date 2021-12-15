import argparse
from collections import defaultdict
import glob
import numpy as np
from os import path as osp
import tqdm

from parse_jsons import get_best_ckpt

def get_dirs(dirpath, regex="*"):
    return list(
        filter(lambda x: osp.isdir(x), glob.glob(osp.join(dirpath, regex)))
    )

def get_metrics(
    json_directory,
    max_key='success',
    silent=False,
    max_ckpt_id=100,
    eval_key=None
):
    ''' Returns a dict mapping json names to a tuple of mean, std, and ckpt ids

    :param json_directory:
    :param silent:
    :return:
    '''
    best_data = defaultdict(list)
    json_queue = []

    for nav_dir in get_dirs(json_directory):
        for seed in range(1, 4):
            for json_dir in get_dirs(nav_dir, regex=f"*seed{seed}*"):
                jkey = (
                    f"{osp.basename(nav_dir)}_"
                    f"{osp.basename(json_dir).split('seed')[0][:-1]}"
                )
                json_queue.append((json_dir, jkey))

    q = tqdm.tqdm(json_queue) if not silent else json_queue

    if eval_key is None:
        eval_key = max_key

    for json_dir, jkey in q:
        id_succ = get_best_ckpt(
            json_dir,
            max_key=max_key,
            silent=True,
            max_ckpt_id=max_ckpt_id,
            val_split="val",
            test_split="test",
            eval_key=eval_key,
        )
        best_data[jkey].append(id_succ)

    # Make sure all have 3 seeds
    means_std = {}
    for k, v in best_data.items():
        # assert len(v) == 3, f"{k} has length {len(v)}, not 3!"

        succ = np.array([x[1] for x in v])
        succ_mean = np.mean(succ)
        # succ_mean = max(succ)
        succ_std = np.std(succ)
        means_std[k] = (succ_mean, succ_std, list(map(lambda x: x[0], v)))

    return means_std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", help="path to dir containing social/pointnav")
    args = parser.parse_args()

    means_std = get_metrics(args.json_dir)


    for i in sorted_m_s:
        print(
            f"{i}:\t{means_std[i][0]*100:.2f},\t{means_std[i][1]*100:.2f},\t{means_std[i][2]}"
        )
