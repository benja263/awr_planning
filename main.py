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

# Internals
from environments.cule_env import CuleEnv
from environments.cule_env_multiple import CuleEnvMultiple
from policies.actor_critic_ts import ActorCriticCnnTSPolicy
from policies.actor_critic_depth0 import ActorCriticCnnPolicyDepth0
from callbacks import WandbTrainingCallback
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
    env_kwargs = dict(env_name=config.env_name, color_mode="gray", repeat_prob=0.0, rescale=True, episodic_life=True,
                      frameskip=4)
    fire_reset = config.env_name not in ["AsterixNoFrameskip-v4", "CrazyClimberNoFrameskip-v4",
                                         "FreewayNoFrameskip-v4", "MsPacmanNoFrameskip-v4",
                                         "SkiingNoFrameskip-v4", "TutankhamNoFrameskip-v4"]
    if config.tree_depth == 0 and config.run_type == "train":
        env = CuleEnvMultiple(env_kwargs=env_kwargs, device="cuda:0",
                              clip_reward=config.clip_reward, fire_reset=fire_reset,
                              n_envs=config.n_envs)
    else:
        env = CuleEnv(env_kwargs=env_kwargs, device=get_device(),
                      clip_reward=config.clip_reward, fire_reset=fire_reset)
    print("Environment:", config.env_name, "Num actions:", env.action_space.n, "Tree depth:", config.tree_depth)

    tensorboard_log = f"./runs/{wandb.run.id}"
    # Setting AWR parameters to the original paper defaults
    awr_def_lr = get_linear_fn(config.actor_lr, 0, 1)
    AWR_params = {"learning_rate": awr_def_lr, "gamma": 0.99, "n_steps": config.n_steps, "batch_size": 256, "normalize_advantage": True,
                  "ent_coef": config.ent_coef, "gae_lambda": 0.95, "policy_gradient_steps": config.policy_gradient_steps, "value_gradient_steps": config.value_gradient_steps, 
                  "learning_starts": config.learning_starts, "value_batch_size": config.value_batch_size, "beta": config.beta,
                  "tensorboard_log": tensorboard_log, 'episodic': config.episodic, "reward_mode": config.reward_mode}

    # Setting AWR models
    if config.tree_depth == 0 and config.run_type == "train":
        model = AWR(policy=ActorCriticCnnPolicyDepth0, env=env, verbose=2, **AWR_params)
    else:        # Hash buffer saves previous states and their trees for reuse in evaluate_actions
        hash_buffer_size = max(config.hash_buffer_size, AWR_params["n_steps"])
        # Input max width sets the maximum number of environments, since the leaves are opened we divide it here to match
        max_width = int(config.max_width / env.action_space.n) if config.max_width != -1 else -1
        policy_kwargs = {"step_env": env, "gamma": config.gamma, "tree_depth": config.tree_depth,
                         "buffer_size": hash_buffer_size, "learn_alpha": config.learn_alpha,
                         "learn_beta": config.learn_beta, "max_width": max_width, "use_leaves_v": config.use_leaves_v, 
                         'hack_optimizer_kwargs': {'actor_lr': config.actor_lr, 'critic_lr': config.critic_lr},
                         "is_cumulative_mode": config.is_cumulative_mode, "regularization": config.regularization}
        AWR_params['buffer_size'] = hash_buffer_size
        model = AWR(policy=ActorCriticCnnTSPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs, **AWR_params)

    # save agent folder and name
    saved_agents_dir = "saved_agents"
    if config.run_type == "train":
        if not os.path.isdir(saved_agents_dir):
            os.makedirs(saved_agents_dir)
        # save agent
        model_filename = "{}/{}".format(saved_agents_dir, wandb.run.id)
        callbacks = [WandbTrainingCallback()]
        model.learn(total_timesteps=config.total_timesteps, log_interval=10, callback=callbacks)
        print("Saving model in " + model_filename)
        model.policy.save(model_filename)
    elif config.run_type == "evaluate":
        if config.model_filename is None:
            raise ValueError("Model filename missing. Please specify using model_filename argument.")
        if config.tree_depth == 0:
            model.policy = ActorCriticCnnPolicyDepth0.load(config.model_filename)
        else:
            model.policy = ActorCriticCnnTSPolicy.load(config.model_filename, lr_schedule=awr_def_lr, env=env)
        avg_score, avg_length = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes,
                                                return_episode_rewards=True, deterministic=False)
        print("Scores of all episodes: ", avg_score)
        print("Lengths of all episodes: ", avg_length)
        print("Average episode score:", np.mean(avg_score), "Average episode length: ", np.mean(avg_length))
    else:
        print("Bad run_type chosen!")


if __name__ == "__main__":
    main()
