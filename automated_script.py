'''
Script to automate stuff.
Makes a new directory, and stores the two yaml files that generate the config.
Replaces the yaml file content with the location of the new directory.
'''

HABITAT_LAB = "/coc/pskynet3/nyokoyama3/learnbycheat/habitat-lab"
RESULTS = "/coc/pskynet3/nyokoyama3/learnbycheat/results_v3"
EXP_YAML  = "habitat_baselines/config/pointnav/behavioral_cloning.yaml"
TASK_YAML = "configs/tasks/pointnav_gibson.yaml"
SLURM_TEMPLATE = "/coc/pskynet3/nyokoyama3/learnbycheat/habitat-lab/slurm_template.sh"
SLURM_TEMPLATE_EVAL = "/coc/pskynet3/nyokoyama3/learnbycheat/habitat-lab/slurm_template_eval.sh"

import os
import argparse
import shutil
import subprocess

parser = argparse.ArgumentParser()
# parser.add_argument('experiment_name')

# Training
parser.add_argument('-n','--num_processes', type=int, default=72)
parser.add_argument('-b','--batch_length', type=int, default=1)
parser.add_argument('-l','--last_teacher_batch', type=float, default=1000)
parser.add_argument('-i','--num_iterations', type=float, default=50000)
parser.add_argument('-s','--sl_lr', type=float, default=2.5e-4)
parser.add_argument('-g','--sgd', action='store_true')
parser.add_argument('-d','--dont_run', action='store_true')
parser.add_argument('-e','--eval', action='store_true')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--delete', action='store_true')
args = parser.parse_args()
args.last_teacher_batch = int(args.last_teacher_batch)
args.num_iterations = int(args.num_iterations)
# experiment_name = args.experiment_name
experiment_name = (
    f'last_{args.last_teacher_batch}_'
    f'batch_{args.batch_length}_'
    f'lr_{args.sl_lr}_'
    f'numproc_{args.num_processes}'
)

if args.sgd:
    experiment_name += '_sgd'

dst_dir = os.path.join(RESULTS, experiment_name)
exp_yaml_path  = os.path.join(HABITAT_LAB, EXP_YAML)
task_yaml_path = os.path.join(HABITAT_LAB, TASK_YAML)
new_task_yaml_path = os.path.join(dst_dir, os.path.basename(task_yaml_path))
new_exp_yaml_path  = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

if not args.eval:
    # Create directory
    if os.path.isdir(dst_dir):
        if args.delete:
            shutil.rmtree(dst_dir)
        else:
            response = input(f"'{dst_dir}' already exists. Delete or abort? [d/A]: ")
            if response == 'd':
                print(f'Deleting {dst_dir}')
                shutil.rmtree(dst_dir)
            else:
                print('Aborting.')
                exit()
    os.mkdir(dst_dir)
    print("Created "+dst_dir)

    # Create task yaml file, using file within Habitat Lab repo as a template
    with open(task_yaml_path) as f:
        task_yaml_data = f.read().splitlines()
    with open(new_task_yaml_path, 'w') as f:
        f.write('\n'.join(task_yaml_data))
    print("Created "+new_task_yaml_path)

    # Create experiment yaml file, using file within Habitat Lab repo as a template
    with open(exp_yaml_path) as f:
        exp_yaml_data = f.read().splitlines()

    for idx, i in enumerate(exp_yaml_data):
        if i.startswith('BASE_TASK_CONFIG_PATH:'):
            exp_yaml_data[idx] = f"BASE_TASK_CONFIG_PATH: '{new_task_yaml_path}'"
        elif i.startswith('TENSORBOARD_DIR:'):
            exp_yaml_data[idx] = f"TENSORBOARD_DIR:    '{os.path.join(dst_dir,'tb')}'"
        elif i.startswith('VIDEO_DIR:'):
            exp_yaml_data[idx] = f"VIDEO_DIR:          '{os.path.join(dst_dir,'video_dir')}'"
        elif i.startswith('EVAL_CKPT_PATH_DIR:'):
            exp_yaml_data[idx] = f"EVAL_CKPT_PATH_DIR: '{os.path.join(dst_dir,'checkpoints')}'"
        elif i.startswith('CHECKPOINT_FOLDER:'):
            exp_yaml_data[idx] = f"CHECKPOINT_FOLDER:  '{os.path.join(dst_dir,'checkpoints')}'"
        elif i.startswith('TXT_DIR:'):
            exp_yaml_data[idx] = f"TXT_DIR:            '{os.path.join(dst_dir,'txts')}'"
        elif i.startswith('  NUM_PROCESSES:'):
            exp_yaml_data[idx] = f"  NUM_PROCESSES: {args.num_processes}"
        elif i.startswith('  BATCH_LENGTH:'):
            exp_yaml_data[idx] = f"  BATCH_LENGTH: {args.batch_length}"
        elif i.startswith('  LAST_TEACHER_BATCH:'):
            exp_yaml_data[idx] = f"  LAST_TEACHER_BATCH: {args.last_teacher_batch}"
        elif i.startswith('  NUM_ITERATIONS:'):
            exp_yaml_data[idx] = f"  NUM_ITERATIONS: {args.num_iterations}"
        elif i.startswith('  SL_LR:'):
            exp_yaml_data[idx] = f"  SL_LR: {args.sl_lr}"
        elif i.startswith('  SGD:'):
            exp_yaml_data[idx] = f"  SGD: {args.sgd}"

    with open(new_exp_yaml_path,'w') as f:
        f.write('\n'.join(exp_yaml_data))
    print("Created "+new_exp_yaml_path)

    # Create slurm job
    with open(SLURM_TEMPLATE) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace('TEMPLATE', experiment_name)
        slurm_data = slurm_data.replace('HABITAT_REPO_PATH', HABITAT_LAB)
        slurm_data = slurm_data.replace('CONFIG_YAML', new_exp_yaml_path)

    slurm_path = os.path.join(dst_dir, experiment_name+'.sh')
    with open(slurm_path,'w') as f:
        f.write(slurm_data)
    print("Generated slurm job: "+slurm_path)

    if not args.dont_run:
        # Submit slurm job
        cmd = 'sbatch '+slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)

        err_file = os.path.join(dst_dir, experiment_name+'.err')
        err_file = err_file.replace('.err','_bc_train.err')
        print(f'\nSee output with:\ntail -F {err_file}')
        print('or')
        print(f'tail -F {err_file}'[:-3]+'out')
else:
    assert args.name != '', 'Must provide experiment name for eval.'
    dst_dir = os.path.join(RESULTS, args.name)
    new_exp_yaml_path  = os.path.join(dst_dir, os.path.basename(exp_yaml_path))

    # Create slurm job
    with open(SLURM_TEMPLATE_EVAL) as f:
        slurm_data = f.read()
        slurm_data = slurm_data.replace('TEMPLATE', args.name)
        slurm_data = slurm_data.replace('HABITAT_REPO_PATH', HABITAT_LAB)
        slurm_data = slurm_data.replace('CONFIG_YAML', new_exp_yaml_path)

    slurm_path = os.path.join(dst_dir, args.name+'_eval.sh')
    with open(slurm_path,'w') as f:
        f.write(slurm_data)
    print("Generated slurm job: "+slurm_path)

    if not args.dont_run:
        # Submit slurm job
        cmd = 'sbatch '+slurm_path
        subprocess.check_call(cmd.split(), cwd=dst_dir)

        err_file = os.path.join(dst_dir, args.name+'_bc_eval.err')
        print(f'\nSee output with:\ntail -F {err_file}')
        print('or')
        print(f'tail -F {err_file}'[:-3]+'out')