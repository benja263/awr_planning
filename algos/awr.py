import torch as th
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from gym import spaces

from typing import Union, List, Any, Dict, Optional, NamedTuple, Type

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from policies.awr_policy import AWRPolicy


ACTIVATION = {'relu': nn.ReLU, 'tanh': nn.Tanh}
OPTIMIZER = {'adam': th.optim.Adam, 'sgd': th.optim.SGD}


class AWR(OffPolicyAlgorithm):

    policy: AWRPolicy
    def __init__(self, 
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            gamma: float = 0.99,
            beta: float = 0.05,
            gae_lambda: float = 0.95,
            ent_coef: float = 0.0,
            weights_max: float = 20,
            learning_starts: int = 10000,
            max_grad_norm: float = 0.0,
            buffer_size: int = 100000,
            episodic: bool = False,
            policy_gradient_steps: int=1000,
            value_gradient_steps: int=250,
            policy_bound_loss_weight: float = 0,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 1,
            value_batch_size: int = 512,
            normalize_advantage: bool = False,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
    ):
        optimizer_class = None
        if policy_kwargs is not None:
            policy_kwargs['activation_fn'] = 'relu' if policy_kwargs is None else ACTIVATION[policy_kwargs.get('activation_fn', 'relu')]
            optimizer_class = policy_kwargs.get('optimizer_class', None)
            if optimizer_class is not None:
                policy_kwargs['optimizer_class'] = OPTIMIZER[optimizer_class]

        self.beta = beta
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.weights_max = weights_max
        self.value_gradient_steps = value_gradient_steps
        self.policy_gradient_steps = policy_gradient_steps
        self.policy_bound_loss_weight = policy_bound_loss_weight
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        try:
            n_envs = env.num_envs
            tr_freq = TrainFreq(n_steps // n_envs, TrainFrequencyUnit.STEP)
        except AttributeError:
            n_envs = 1
            if episodic:
                tr_freq = TrainFreq(n_steps, TrainFrequencyUnit.EPISODE)
            else:
                tr_freq = TrainFreq(n_steps // n_envs, TrainFrequencyUnit.STEP)
                
        super().__init__(policy=policy,
        env=env,
        policy_base=None,
        replay_buffer_class=AWRReplayBuffer,
        support_multi_env=True if n_envs > 1 else False,
        tensorboard_log=tensorboard_log,
        seed=seed,
        train_freq = tr_freq,
        verbose=verbose,
        device=device,
        learning_starts=learning_starts,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        replay_buffer_kwargs={'gae_lambda': gae_lambda, 'gamma': gamma},
        )
        super()._setup_model()


        self.bound_min = self.get_action_bound_min()
        self.bound_max = self.get_action_bound_max()
        self.epochs = 0
        self.value_batch_size = value_batch_size


    def get_values(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        n_samples, n_envs = observations.shape[0], observations.shape[1]
        values = np.zeros((n_samples, n_envs), dtype=np.float32)
        next_values = np.zeros((n_samples, n_envs), dtype=np.float32)
        # print(f"n_samples: {n_samples}, batch_size: {batch_size}")
        for env_idx in range(n_envs):
            for i in range(0, n_samples, self.value_batch_size):
                # print(f"i: {i}")
                batch_obs = self.replay_buffer.to_torch(observations[i:i+self.value_batch_size, env_idx])
                batch_next_obs = self.replay_buffer.to_torch(next_observations[i:i+self.value_batch_size, env_idx])
                # print(f"obs shape: {batch_obs.shape}")
                torch_values = self.policy.predict_values(batch_obs)
                torch_next_values = self.policy.predict_values(batch_next_obs)
                
                values[i:i+self.value_batch_size, env_idx] = torch_values.detach().cpu().numpy().squeeze()
                next_values[i:i+self.value_batch_size, env_idx] = torch_next_values.detach().cpu().numpy().squeeze()
        return values, next_values

    def get_action_bound_min(self):
        if (isinstance(self.action_space, spaces.Box)):
            bound_min = self.action_space.low
        else:
            bound_min = -np.inf * np.ones(1)
        return th.tensor(bound_min, device=self.device)

    def get_action_bound_max(self):
        if (isinstance(self.action_space, spaces.Box)):
            bound_max = self.action_space.high
        else:
            bound_max = np.inf * np.ones(1)
        return th.tensor(bound_max, device=self.device)
  
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
         # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # print('training')
        env = self.env if isinstance(self.env, VecNormalize) else None 
        observations = self.replay_buffer._normalize_obs(self.replay_buffer.observations, env)
        # observations = self.replay_buffer.to_torch(observations)
        next_observations = self.replay_buffer._normalize_obs(self.replay_buffer.next_observations, env)
        # next_observations = self.replay_buffer.to_torch(next_observations)
        # print(f"replay buffer: {self.replay_buffer.valid_pos}, pos: {self.replay_buffer.pos}")
        observations = observations[:self.replay_buffer.valid_pos]
        next_observations = next_observations[:self.replay_buffer.valid_pos]

        values, next_values = self.get_values(observations, next_observations)
        self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)

        value_losses = []
        for gradient_step in range(self.value_gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            pred_values = self.policy.predict_values(replay_data.observations)

            value_loss = 0.5*F.mse_loss(pred_values.squeeze(), replay_data.returns.squeeze())
            self.policy.optimizer.zero_grad()
            value_loss.backward()
            if self.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizer.step()
            value_losses.append(value_loss.item())


        policy_losses = []
        values, next_values = self.get_values(observations, next_observations)
        self.replay_buffer.add_advantages_returns(values, next_values, env=self._vec_normalize_env)
        self._n_updates += self.value_gradient_steps 

        for gradient_step in range(self.policy_gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            actions = replay_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            advantages = replay_data.advantages

            if self.normalize_advantage:
                adv_mean, adv_std = advantages.mean(), advantages.std() 
                advantages = (advantages - adv_mean) / (adv_std + 1e-5)

            weights = th.exp(advantages / self.beta)
            weights = th.clamp(weights, max=self.weights_max)

            values, log_prob, entropy = self.policy.evaluate_actions(replay_data.observations, actions)

            policy_loss = -(log_prob*weights).mean() - self.ent_coef*th.mean(entropy)

            # if self.policy_bound_loss_weight > 0 and isinstance(self.action_space, spaces.Box):
            #     distrib = self.policy.actor.get_distribution(replay_data.observations)
            #     val = distrib.mode()
            #     vio_min = th.clamp(val - self.bound_min, max=0)
            #     vio_max = th.clamp(val - self.bound_max, min=0)
            #     violation = vio_min.pow_(2).sum(axis=-1) + vio_max.pow_(2).sum(axis=-1)
            #     policy_loss += 0.5 * th.mean(violation)

            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            if self.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            policy_losses.append(policy_loss.item())

        self._n_updates += self.policy_gradient_steps
        self.epochs += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/replay_buffer_pos", self.replay_buffer.pos)
        self.logger.record("train/replay_buffer_size", self.replay_buffer.buffer_size)
        self.logger.record("train/replay_buffer_full", self.replay_buffer.full)
        self.logger.record("train/policy_loss", np.mean(policy_losses))


    def learn(
        self,
        total_timesteps: int,
        callback= None,
        log_interval: int = 100,
        tb_log_name: str = "",
        reset_num_timesteps: bool = True,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
        )
    

class AWRReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class AWRReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like AWR.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma: float, 
        gae_lambda: float,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=False, handle_timeout_termination=True)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.gamma = gamma 
        self.gae_lambda = gae_lambda

        self.valid_pos = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        if isinstance(obs, th.Tensor):
            obs = obs.cpu()
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        if isinstance(action, th.Tensor):
            action = action.cpu()
        if isinstance(reward, th.Tensor):
            reward = reward.cpu()
        if isinstance(done, th.Tensor):
            done = done.cpu()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        self.valid_pos = self.buffer_size if self.full else self.pos

    
    def add_advantages_returns(self, values: np.array, next_values: np.array, env: Optional[VecNormalize] = None) -> None:

        """Compute Lambda return"""
        assert len(values) == self.valid_pos 
        self.advantages = np.zeros((self.valid_pos, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.valid_pos, self.n_envs), dtype=np.float32)
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)
        for env_id in range(self.n_envs):
            for step in reversed(range(self.valid_pos)):
                rewards = self._normalize_reward(self.rewards[:self.valid_pos, env_id].reshape(-1, 1), env)
                non_terminal = 1.0 - self.dones[step, env_id] * (1.0 - self.timeouts[step, env_id])
                q_est = rewards[step] + self.gamma * next_values[step, env_id] * non_terminal
                self.returns[step, env_id] = q_est + self.gamma * self.gae_lambda * non_terminal * last_gae_lam[env_id]
                self.advantages[step, env_id] = self.returns[step, env_id] - values[step, env_id]
                last_gae_lam[env_id] = self.returns[step, env_id] - values[step, env_id]

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> AWRReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> AWRReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.advantages[batch_inds, env_indices],
            self.returns[batch_inds, env_indices],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return AWRReplayBufferSamples(*tuple(map(self.to_torch, data)))

