#!/bin/bash
## wandb parameters
wandb_username='nvidia-mellanox'
wandb_api_key="99bff6368b8da60bf901568aa87c53db19adebcc"

project_name="awr_planning" #"stable_baselines3-quantiles" #"pg-tree"
git_repo="." #"pg-tree"
naming="debug"
sweep_id="spaor26i" 
#additional_commands="ssh-keyscan github.com >> ~/.ssh/known_hosts && git fetch && git checkout test_new_alpha && git pull"
# additional_commands="git pull"
additional_commands="cd awr_planning && git pull"
#additional_commands="ssh-keyscan github.com >> ~/.ssh/known_hosts && git checkout quantiles && git pull && pip install sortedcontainers"


## ngc parameters
num_agents=1
# 16, 32, 40
instance="dgx1v.16g.1.norm"
# dgx1v.32g.1.norm x32
# dgx1v.16g.1.norm x32
# dgxa100.40g.1.norm x8
# ovxa40.48g.1 x64
# dgxa100.20g.1.norm.mig.3
dataset_id=0
workspace="ws-benjamin:test_awr"
image='nvcr.io/nvidian/nbu-ai/awr_stm'

label="_wl___reinforcement_learning"

# project parameters
# sub_dir="docker_internal" #'none', 'docker_internal' # subdirectory from which we want to run the agent, 'none' if not needed
sub_dir="docker_internal" #'none', 'docker_internal' # subdirectory from which we want to run the agent, 'none' if not needed
git_repo_url="https://github.com/benja263/awr_planning.git" # includes special identifier so git would not ask for password


# run command
mkdir -p logs
python3 generate_ngc_run.py --image $image --workspace $workspace --project_name $project_name --instance $instance --dataset_id $dataset_id \
 --num_agents $num_agents --label $label --sweep_id $sweep_id --git_repo $git_repo --sub_dir $sub_dir --wandb_api_key $wandb_api_key \
  --naming $naming --git_repo_url $git_repo_url --wandb_username $wandb_username --additional_commands "$additional_commands" 2>&1 | tee ./logs/generate_ngc_runs.log

