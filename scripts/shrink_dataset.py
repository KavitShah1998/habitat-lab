import json
import gzip
import glob
import os
import os.path as osp

val = "data/spot_goal_headings_hm3d/val_orig/content"
out_content = "data/spot_goal_headings_hm3d/val/content"

os.makedirs(out_content, exist_ok=True)

files = glob.glob(osp.join(val, "*json.gz"))

for file in files:
    with gzip.open(file) as f:
        data = json.loads(f.read().decode())
    data['episodes'] = data['episodes'][:20]

    with gzip.open(file.replace(val, out_content), "wt") as f:
        json.dump(data, f)
