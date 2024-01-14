import argparse
import torch
import torch.backends.cudnn
import numpy as np
import random

from collections import deque

class DequeDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self.d = {}
        self.keys = deque()

    def __setitem__(self, key, value):
        if key not in self.d:
            if len(self.keys) == self.max_size:
                oldest_key = self.keys.popleft()
                del self.d[oldest_key]
            self.keys.append(key)
        self.d[key] = value

    def __getitem__(self, key):
        return self.d[key]

    def __delitem__(self, key):
        self.d.pop(key)
        self.keys.remove(key)

    def __contains__(self, key):
        return key in self.d

    def get(self, key, default=None):
        return self.d.get(key, default)

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def __repr__(self):
        return repr(self.d)
    

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def groupby_sum(value, labels) -> (torch.Tensor, torch.LongTensor):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, value)
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result, new_labels

def add_regularization_logits(logits, epsilon):
    A = logits.shape[1]
    probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
    new_probs = (1-epsilon) * probs + epsilon/A
    new_logits = torch.log(new_probs)
    return  new_logits


def create_parser():
    # TODO: Add help
    # TODO: remove irrelevant parameters?
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=200000000,
                        help="Number of environment steps for training")
    parser.add_argument("--actor_lr", type=float, default=5.0e-5, help="Optimizer learning rate")
    parser.add_argument("--critic_lr", type=float, default=1.0e-4, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=4, help="Seed for all pseudo-random generators")
    parser.add_argument("--env_name", type=str, default="AlienNoFrameskip-v4", help="Environment name")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--noop_max", type=int, default=30, help="noop max")
    parser.add_argument("--ent_coef", type=float, default=0., help="entropy coefficient")
    parser.add_argument("--tree_depth", type=int, default=0, help="SoftTreeMax depth (0 corresponds to standard PPO)")
    parser.add_argument("--clip_reward", type=str2bool, nargs="?", const=True, default=True,
                        help="Reward clipping wrapper")
    parser.add_argument("--learn_alpha", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to treat alpha (weight of root value) as a learnable parameter or constant")
    parser.add_argument("--learn_beta", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to treat beta (temperature parameter) as a learnable parameter or constant")
    parser.add_argument("--episodic", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to treat beta (temperature parameter) as a learnable parameter or constant")
    parser.add_argument("--max_width", type=int, default=-1,
                        help="Maximal SoftTreeMax width, beyond which the tree will be truncated. "
                             "Use -1 to not limit width.")
    parser.add_argument("--experiment_type", type=str, default="", help="Free text to describe experiment goal")
    # experiment_type examples: Runtime_optimization, Debug, Paper_main, Ablation, Hyperparameter_sweep
    parser.add_argument("--experiment_description", type=str, default="",
                        help="Free text to describe experiment sub-goal")
    parser.add_argument("--hash_buffer_size", type=int, default=1000, help="Size of buffer which stores leaf values")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Size of buffer which stores leaf values")
    parser.add_argument("--use_leaves_v", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use the value at the leaves or reward only")
    parser.add_argument("--is_cumulative_mode", type=str2bool, nargs="?", const=True, default=False,
                        help="True for Cumulative SoftTreeMax. False for Exponentiated SoftTreeMax")
    parser.add_argument("--regularization", type=float, default=0.001, help="Minimal probability for all actions")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel AWR environments on GPU")
    parser.add_argument("--n_steps", type=int, default=4, help="Number of steps in the environment per rollout")
    parser.add_argument("--value_batch_size", type=int, default=32, help="Batch size used to calculate values")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used")
    parser.add_argument("--learning_starts", type=int, default=100000, help="Number of warm-up iterations")
    parser.add_argument("--beta", type=float, default=1, help="AWR beta parameter")
    parser.add_argument("--reward_mode", type=str, default="monte-carlo", help="AWR reward mode calculation", choices=['monte-carlo', 'gae'])
    # Evaluation fields
    parser.add_argument("--run_type", type=str, default="train", help="Train or evaluate")  # train or evaluate
    parser.add_argument("--model_filename", type=str, default=None, help="Filename to store or load model")
    parser.add_argument("--n_eval_episodes", type=int, default=200, help="Number of evaluation episodes")
    parser.add_argument("--value_gradient_steps", type=int, default=1, help="Number of gradeient steps for value function")
    parser.add_argument("--policy_gradient_steps", type=int, default=1, help="Number of gradeient steps for policy")
    return parser
