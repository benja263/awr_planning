# Built-ins
import sys
import os

# For NGC runs, TODO: Remove this in final version
# sys.path.append("../stable-baselines3/")

# Externals
import wandb
import numpy as np
from algos.awr import AWR
from stable_baselines3.common.utils import get_device, get_linear_fn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from policies.awr_policy import AWRPolicy
from wandb.integration.sb3 import WandbCallback
from utils import create_parser, set_seed
from stable_baselines3.common.logger import configure
# 

# from wandb.integration.sb3 import WandbCallback
if sys.gettrace() is not None:
    os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_BASE_URL"] = "http://api.wandb.ai"


def main():
    # Input arguments
    parser = create_parser()
    wandb.init(config=parser.parse_args(), project="pg-tree", monitor_gym=True, sync_tensorboard=True)
    config = wandb.config
    set_seed(config.seed)
    # Setting environment
    env = make_vec_env(config.env_name, n_envs=config.n_envs, seed=config.seed)
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False)
    print("Environment:", config.env_name)

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=None,
        verbose=1)
    tensorboard_log = f"./runs/{wandb.run.id}"

    # Setting PPO parameters to the original paper defaults
    awr_def_lr = get_linear_fn(config.actor_lr, 0, 1)
    AWR_params = {"learning_rate": awr_def_lr, "gamma": 0.99, "n_steps": config.n_steps, "batch_size": 256, "normalize_advantage": True,
                  "ent_coef": config.ent_coef, "gae_lambda": 0.95, "policy_gradient_steps": config.policy_gradient_steps, "value_gradient_steps": config.value_gradient_steps, 
                  "learning_starts": 10000, "value_batch_size": config.value_batch_size, "beta": config.beta, "learning_starts": 1000,
                  "tensorboard_log": tensorboard_log, "reward_mode": config.reward_mode,
                'episodic': config.episodic, "policy_kwargs": {'hack_optimizer_kwargs': {'actor_lr': config.actor_lr, 'critic_lr': config.critic_lr}} }
    # Setting PPO models
    model = AWR(policy=AWRPolicy, env=env, verbose=2, **AWR_params)

    # save agent folder and name
    saved_agents_dir = "saved_agents"
    if config.run_type == "train":
        if not os.path.isdir(saved_agents_dir):
            os.makedirs(saved_agents_dir)
        # save agent
        model_filename = "{}/{}".format(saved_agents_dir, wandb.run.id)
        model.learn(total_timesteps=config.total_timesteps, log_interval=10, callback=wandb_callback)
        print("Saving model in " + model_filename)
        model.policy.save(model_filename)
    elif config.run_type == "evaluate":
        if config.model_filename is None:
            raise ValueError("Model filename missing. Please specify using model_filename argument.")
        model.policy = ActorCriticPolicy.load(config.model_filename)
        avg_score, avg_length = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes,
                                                return_episode_rewards=True, deterministic=False)
        print("Scores of all episodes: ", avg_score)
        print("Lengths of all episodes: ", avg_length)
        print("Average episode score:", np.mean(avg_score), "Average episode length: ", np.mean(avg_length))
    else:
        print("Bad run_type chosen!")


if __name__ == "__main__":
    main()
