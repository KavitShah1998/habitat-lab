import argparse
import tqdm
import glob
import json
import numpy as np

from os import path as osp

VALIDATION_SCENE_IDS = [4, 0, 3, 5]
val_eps = []
for s_id in VALIDATION_SCENE_IDS:
    val_eps.extend(list(range(s_id * 71, (s_id + 1) * 71)))
test_eps = list(filter(lambda x: x not in val_eps, range(994)))
splits = {
    "val": val_eps,
    "test": test_eps,
    "all": val_eps+test_eps,
}

TIME_FILE = '/private/home/naokiyokoyama/qq/rrt_star/icra.json'
with open(TIME_FILE) as f:
    EP2TIME = json.load(f)
STEP_PERIOD = 0.1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    args = parser.parse_args()

    json_dir = args.json_dir
    if json_dir.endswith('/'):
        json_dir = json_dir[:-1]

    best_ckpt_id, test_set_succ = get_best_ckpt(json_dir)
    ckpt_path = osp.realpath(
        osp.join(
            osp.dirname(osp.dirname(osp.dirname(json_dir))),
            f'checkpoints/{osp.basename(json_dir)}/ckpt.{best_ckpt_id}.pth'
        )
    )
    print(f"Best ckpt ID: {best_ckpt_id} ({ckpt_path})")
    print(f"Test set avg succ: {test_set_succ * 100:.2f}%")


# Get mean value filtered by split
def get_mean_val(key, stats, eps):
    use_sct = key == 'sct'

    def steps2sct(ep_id, v, steps):
        succ = v['success']
        comp_time = EP2TIME[str(ep_id)]
        agent_time = steps*STEP_PERIOD
        sct = succ * (comp_time/max(agent_time, comp_time))

        return sct

    if not use_sct:
        mean = np.mean(
            [
                v[key]
                for k, v in stats.items()
                if k != "agg_stats" and int(k) in eps
            ]
        )
    else:
        mean = np.mean(
            [
                steps2sct(k, v, v['num_steps'])
                for k, v in stats.items()
                if k != "agg_stats" and int(k) in eps
            ]
        )
    return mean


def get_best_ckpt(
    json_dir,
    max_key="success",
    silent=False,
    max_ckpt_id=None,
    val_split="val",
    test_split="test",
    eval_key=None,
):

    # Gather paths to all json files
    json_paths = glob.glob(osp.join(json_dir, "*.json"))

    # Dict of dicts
    all_j_data = {}

    # Load data from all json files
    if not silent:
        print("Parsing all jsons...")
        paths = tqdm.tqdm(json_paths)
    else:
        paths = json_paths
    for j_path in paths:
        # Load from JSON file
        with open(j_path) as f:
            stats = json.load(f)

        ckpt_id = int(j_path.split("_")[-1].split(".")[0])
        if max_ckpt_id is not None and ckpt_id > max_ckpt_id:
            continue
        all_j_data[ckpt_id] = stats

    # Identify best checkpoint
    all_succ_vals = [
        (get_mean_val(max_key, stats, splits[val_split]), ckpt_id)
        for ckpt_id, stats in all_j_data.items()
    ]

    best_ckpt_id = max(all_succ_vals)[1]

    if eval_key is None:
        eval_key = max_key

    # Use best checkpoint to calculate avg succ on test set
    test_set_succ = get_mean_val(
        eval_key, all_j_data[best_ckpt_id], splits[test_split]
    )

    return best_ckpt_id, test_set_succ


if __name__ == "__main__":
    main()
