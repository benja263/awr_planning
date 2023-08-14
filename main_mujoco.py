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

from stable_baselines3.common.policies import ActorCriticPolicy
from wandb.integration.sb3 import WandbCallback
from utils import create_parser, set_seed

# from wandb.integration.sb3 import WandbCallback
if sys.gettrace() is not None:
    os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_BASE_URL"] = "http://api.wandb.ai"


def main():
    # Input arguments
    parser = create_parser()
    wandb.init(config=parser.parse_args(), project="pg-tree")
    config = wandb.config

    set_seed(config.seed)
    # Setting environment
    env = make_vec_env(config.env_name, n_envs=config.n_envs, seed=config.seed)
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False)
    print("Environment:", config.env_name)

    # Setting PPO parameters to the original paper defaults
    awr_def_lr = get_linear_fn(config.learning_rate, 0, 1)
    AWR_params = {"learning_rate": awr_def_lr, "gamma": 0.99, "n_steps": 2048, "batch_size": 32, "normalize_advantage": True,
                  "ent_coef": 0.01, "gae_lambda": 0.95, "policy_gradient_steps": 1000, "value_gradient_steps": 200, 
                  "learning_starts": 10000, "value_batch_size": config.value_batch_size, "beta": config.beta}

    # Setting PPO models
    model = AWR(policy=ActorCriticPolicy, env=env, verbose=2, **AWR_params)


    # save agent folder and name
    saved_agents_dir = "saved_agents"
    if config.run_type == "train":
        if not os.path.isdir(saved_agents_dir):
            os.makedirs(saved_agents_dir)
        # save agent
        model_filename = "{}/{}".format(saved_agents_dir, wandb.run.id)
        callbacks = [WandbCallback()]
        model.learn(total_timesteps=config.total_timesteps, log_interval=None, callback=callbacks)
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
