# Built-in
from typing import Tuple
import math

# Externals
import torch as th
from stable_baselines3.common.utils import get_device

# Internals
from policies.actor_critic_depth0 import ActorCriticCnnPolicyDepth0
from policies.cule_bfs import CuleBFS
from utils import add_regularization_logits, DequeDict


class ActorCriticCnnTSPolicy(ActorCriticCnnPolicyDepth0):
    def __init__(self, observation_space, action_space, lr_schedule, tree_depth, gamma, step_env, buffer_size,
                 learn_alpha, learn_beta, max_width, use_leaves_v, is_cumulative_mode, regularization, **kwargs):
        super(ActorCriticCnnTSPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.cule_bfs = CuleBFS(step_env, tree_depth, gamma, self.compute_value, max_width)
        self.time_step = 0
        self.obs2leaves_dict = DequeDict(max_size=buffer_size)
        self.timestep2obs_dict = DequeDict(max_size=buffer_size)
        self.obs2timestep_dict = DequeDict(max_size=buffer_size)
        self.buffer_size = buffer_size
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.tree_depth = tree_depth
        self.max_width = max_width
        self.is_cumulative_mode = is_cumulative_mode
        self.regularization = regularization
        self.alpha = th.tensor(0.5 if learn_alpha else 1.0, device=self.device)
        self.beta = th.tensor(1.0, device=self.device)
        self.lr_schedule = lr_schedule
        if self.learn_alpha:
            self.alpha = th.nn.Parameter(self.alpha)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        if self.learn_beta:
            self.beta = th.nn.Parameter(self.beta)
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.use_leaves_v = use_leaves_v

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        hash_obs = self.hash_obs(obs)[0].item()
        if hash_obs in self.obs2leaves_dict:
            leaves_observations, rewards, first_action = self.obs2leaves_dict.get(hash_obs)
            # print(f'already inside leaves_observations.shape: {leaves_observations.shape}, rewards.shape: {rewards.shape}, first_action.shape: {first_action.shape if first_action is not None else None} ')
            # leaves_observations, rewards, first_action = leaves_observations.to(obs.device), rewards.to(obs.device), first_action if first_action is None else first_action.to(obs.device)
            # if hash_obs in self.timestep2obs_dict:
            #     del self.timestep2obs_dict[self.obs2timestep_dict[hash_obs]]
        else:
            leaves_observations, rewards, first_action = self.cule_bfs.bfs(obs, self.cule_bfs.max_depth)
            # print(f'new obs.shape {obs.shape} leaves_observations.shape: {leaves_observations.shape}, rewards.shape: {rewards.shape}, first_action.shape: {first_action.shape if first_action is not None else None} ')
            # self.obs2leaves_dict[hash_obs] = leaves_observations.cpu(), rewards.cpu(), first_action if first_action is None else first_action.cpu()
            self.obs2leaves_dict[hash_obs] = leaves_observations, rewards, first_action
        self.obs2timestep_dict[hash_obs] = self.time_step
        self.timestep2obs_dict[self.time_step] = hash_obs
        # Preprocess the observation if needed
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        if self.use_leaves_v:
            latent_pi, value_root = self.compute_value(leaves_obs=leaves_observations, root_obs=obs)
            value_root = (val_coef * value_root + rewards.reshape([-1, 1])).max()
        else:
            latent_pi, value_root = self.compute_value_with_root(leaves_obs=leaves_observations, root_obs=obs)
        mean_actions = val_coef * self.actor.action_net(latent_pi) + rewards.reshape([-1, 1])
        if self.cule_bfs.max_width == -1:
            mean_actions_per_subtree = self.beta * mean_actions.reshape([self.action_space.n, -1])
            counts = th.ones([1, self.action_space.n]) * mean_actions_per_subtree.shape[1]
        else:
            mean_actions_per_subtree = th.zeros(self.action_space.n, mean_actions.shape[0], mean_actions.shape[1],
                                                device=mean_actions.device)
            idxes = th.arange(mean_actions.shape[0])
            counts = th.zeros(self.action_space.n)
            v, c = th.unique(first_action, return_counts=True)
            # print(counts.device, c.device)
            # counts[v] = (c.type(th.float32) * self.action_space.n).cpu()
            counts[v] = c.type(th.float32) * self.action_space.n
            mean_actions_per_subtree[first_action.flatten(), idxes, :] = mean_actions
            mean_actions_per_subtree = self.beta * mean_actions_per_subtree.reshape([self.action_space.n, -1])
        counts = counts.to(mean_actions.device).reshape([1, -1])
        if self.is_cumulative_mode:
            mean_actions_logits = th.sum(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) / counts
        else:
            # To obtain the mean we subtract the normalization log(#leaves)
            mean_actions_logits = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) - \
                                  th.log(counts)
        mean_actions_logits[counts == 0] = -math.inf
        depth0_logits = self.compute_value(leaves_obs=obs)[0] if self.learn_alpha else th.tensor(0)
        if th.any(th.isnan(mean_actions_logits)):
            print("NaN in forward:mean_actions_logits.")
            mean_actions_logits[th.isnan(mean_actions_logits)] = 0
        if th.any(th.isnan(depth0_logits)):
            print("NaN in forward:depth0_logits.")
            depth0_logits[th.isnan(depth0_logits)] = 0
        mean_actions_logits = self.alpha * mean_actions_logits + (1 - self.alpha) * depth0_logits
        mean_actions_logits = add_regularization_logits(mean_actions_logits, self.regularization)
        distribution = self.actor.action_dist.proba_distribution(action_logits=mean_actions_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # if self.time_step - self.buffer_size in self.timestep2obs_dict:
        #     if self.timestep2obs_dict[self.time_step - self.buffer_size] in self.obs2leaves_dict: 
        #         del self.obs2leaves_dict[self.timestep2obs_dict[self.time_step - self.buffer_size]]
        #     if self.timestep2obs_dict[self.time_step - self.buffer_size] in self.obs2timestep_dict: 
        #         del self.obs2timestep_dict[self.timestep2obs_dict[self.time_step - self.buffer_size]]
        #     del self.timestep2obs_dict[self.time_step - self.buffer_size]
        self.time_step += 1
        # print(f"actions: {actions} shape: {actions.shape}, value_root: {value_root}")
        return actions, value_root, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        self.add_gradients_history()
        batch_size = obs.shape[0]
        mean_actions_logits = th.zeros((batch_size, self.action_space.n), device=actions.device)
        ret_values = th.zeros((batch_size, 1), device=actions.device)
        # Preprocess the observation if needed
        hash_obses = self.hash_obs(obs)
        all_leaves_obs = [] if self.use_leaves_v else [obs]
        all_rewards = []
        all_first_actions = []
        for i in range(batch_size):
            hash_obs = hash_obses[i].item()
            if hash_obs in self.obs2leaves_dict:
                leaves_observations, rewards, first_action = self.obs2leaves_dict.get(hash_obs)
                # leaves_observations, rewards, first_action = leaves_observations.to(obs.device), rewards.to(obs.device),  first_action if first_action is None else first_action.to(obs.device)
            else:
                #print("This should not happen! observation not in our dictionary")
                leaves_observations, rewards, first_action = self.cule_bfs.bfs(obs[i], self.cule_bfs.max_depth)
                self.obs2leaves_dict[hash_obs] = leaves_observations, rewards, first_action
                # self.obs2leaves_dict[hash_obs] = leaves_observations.cpu(), rewards.cpu(), first_action if first_action is None else first_action.cpu()
            all_leaves_obs.append(leaves_observations)
            all_rewards.append(rewards)
            all_first_actions.append(first_action)
            # Preprocess the observation if needed
        all_rewards_th = th.cat(all_rewards).reshape([-1, 1])
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        cat_features = th.cat(all_leaves_obs)
        if self.use_leaves_v:
            values = self.critic(cat_features)
            latent_pi = self.actor.get_latent_pi(cat_features)
        else:
            latent_pi = self.actor.get_latent_pi(cat_features[batch_size:])
            values = self.critic(cat_features[:batch_size])
        # Assaf added
        mean_actions = val_coef * self.actor.action_net(latent_pi) + all_rewards_th
        if self.use_leaves_v:
            values_subtrees = val_coef * values + all_rewards_th
        subtree_width = self.action_space.n ** self.cule_bfs.max_depth
        if self.cule_bfs.max_width != -1:
            subtree_width = min(subtree_width, self.cule_bfs.max_width*self.action_space.n)
        for i in range(batch_size):
            mean_actions_batch = mean_actions[subtree_width * i:subtree_width * (i + 1)]
            if self.use_leaves_v:
                ret_values[i, 0] = values_subtrees[subtree_width * i:subtree_width * (i + 1)].max()
            if self.cule_bfs.max_width == -1:
                subtree_width = self.action_space.n ** self.cule_bfs.max_depth
                mean_actions_per_subtree = self.beta * mean_actions_batch.reshape([self.action_space.n, -1])
                counts = th.ones([1, self.action_space.n]) * mean_actions_per_subtree.shape[1]
            else:
                mean_actions_per_subtree = th.zeros(self.action_space.n, mean_actions_batch.shape[0], mean_actions_batch.shape[1],
                                                    device=mean_actions_batch.device) # - 1e6
                idxes = th.arange(mean_actions_batch.shape[0])
                counts = th.zeros(self.action_space.n)
                v, c = th.unique(all_first_actions[i], return_counts=True)
                # counts[v] = (c.type(th.float32) * self.action_space.n).cpu()
                counts[v] = c.type(th.float32) * self.action_space.n
                mean_actions_per_subtree[all_first_actions[i].flatten(), idxes, :] = mean_actions_batch
                mean_actions_per_subtree = self.beta * mean_actions_per_subtree.reshape([self.action_space.n, -1])
            counts = counts.to(mean_actions.device).reshape([1, -1])
            if self.is_cumulative_mode:
                mean_actions_logits[i, :] = th.sum(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) / counts
            else:
                mean_actions_logits[i, :] = th.logsumexp(mean_actions_per_subtree, dim=1, keepdim=True).transpose(1, 0) - \
                                  th.log(counts)
            mean_actions_logits[i, counts[0, :] == 0] = -math.inf

        depth0_logits = self.compute_value(leaves_obs=obs)[0] if self.learn_alpha else th.tensor(0)
        if th.any(th.isnan(mean_actions_logits)):
            print("NaN in eval_actions:mean_actions_logits!!!")
            mean_actions_logits[th.isnan(mean_actions_logits)] = 0
        if th.any(th.isnan(depth0_logits)):
            print("NaN in eval_actions:depth0_logits!!!")
            depth0_logits[th.isnan(depth0_logits)] = 0
        mean_actions_logits = self.alpha * mean_actions_logits + (1 - self.alpha) * depth0_logits
        mean_actions_logits = add_regularization_logits(mean_actions_logits, self.regularization)
        distribution = self.actor.action_dist.proba_distribution(action_logits=mean_actions_logits)
        log_prob = distribution.log_prob(actions)
        if self.use_leaves_v:
            return log_prob, distribution.entropy()
        else:
            return log_prob, distribution.entropy()

    def hash_obs(self, obs):
        return (obs[:, -2:, :, :].int()).view(obs.shape[0], -1).sum(dim=1)

    def compute_value_with_root(self, leaves_obs, root_obs=None):
        if root_obs is None:
            return self.actor.get_mean_actions(leaves_obs), None
       
        cat_features = th.cat((root_obs, leaves_obs))
        # print(cat_features.shape,cat_features[:1].shape, cat_features[1:].shape )
        latent_pi = self.actor.get_latent_pi(cat_features[1:])
        value_root = self.predict_values(cat_features[:1])
        return latent_pi, value_root

    def compute_value(self, leaves_obs, root_obs=None):
        if root_obs is None:
            return self.actor.get_mean_actions(leaves_obs), None
        value_root = self.predict_values(leaves_obs)
        latent_pi = self.actor.get_latent_pi(leaves_obs)
        return latent_pi, value_root
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor: 
        """
        Forward pass in critic only

        :param obs: Observation
        :return: value
        """
        # print(f" buffer size: {self.buffer_size} len(self.obs2leaves_dict): {len(self.obs2leaves_dict)} len(self.obs2timestep_dict): {len(self.obs2timestep_dict)} len(self.timestep2obs_dict): {len(self.timestep2obs_dict)}")
        # print(f"predict_values: obs.shape: {obs.shape}")
        batch_size = obs.shape[0]
        ret_values = th.zeros((batch_size, 1), device=obs.device)
        hash_obses = self.hash_obs(obs)
        all_leaves_obs = [] if self.use_leaves_v else [obs]
        all_rewards = []
        for i in range(batch_size):
            hash_obs = hash_obses[i].item()
            if hash_obs in self.obs2leaves_dict:
                leaves_observations, rewards, _ = self.obs2leaves_dict.get(hash_obs)
                # leaves_observations, rewards = leaves_observations.to(obs.device), rewards.to(obs.device)
            else:
                leaves_observations, rewards, first_action = self.cule_bfs.bfs(obs[i], self.cule_bfs.max_depth)
                # first_action = first_action if first_action is None else first_action.cpu()
                # self.obs2leaves_dict[hash_obs] = leaves_observations.cpu(), rewards.cpu(), first_action if first_action is None else first_action.cpu()
                self.obs2leaves_dict[hash_obs] = leaves_observations, rewards, first_action
                self.obs2timestep_dict[hash_obs] = self.time_step
                self.timestep2obs_dict[self.time_step] = hash_obs
                self.time_step += 1
            all_leaves_obs.append(leaves_observations)
            all_rewards.append(rewards)
        all_rewards_th = th.cat(all_rewards).reshape([-1, 1])
        val_coef = self.cule_bfs.gamma ** self.cule_bfs.max_depth
        # print(f"obs shape: {obs.shape} th.cat(all_leaves_obs, dim=0).shape: {th.cat(all_leaves_obs, dim=0).shape}")
        cat_features = th.cat(all_leaves_obs, dim=0)
        if self.use_leaves_v:
            values = self.critic(cat_features)
        else:
            values = self.critic(cat_features[:batch_size])
        if self.use_leaves_v:
            values_subtrees = val_coef * values + all_rewards_th
        else:
            return values 
        subtree_width = self.action_space.n ** self.cule_bfs.max_depth
        if self.cule_bfs.max_width != -1:
            subtree_width = min(subtree_width, self.cule_bfs.max_width*self.action_space.n)
        for i in range(batch_size):
            if self.use_leaves_v:
                ret_values[i, 0] = values_subtrees[subtree_width * i:subtree_width * (i + 1)].max()
            return ret_values

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path: (str)
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_data(),
                 "alpha": self.alpha, "beta": self.beta, "time_step": self.time_step}, path)

    def _get_data(self):
        """
        Get data that need to be saved in order to re-create the model.
        This corresponds to the arguments of the constructor.

        :return: (Dict[str, Any])
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            tree_depth=self.tree_depth,
            gamma=self.cule_bfs.gamma,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            buffer_size=self.buffer_size,
            learn_alpha=self.learn_alpha,
            learn_beta=self.learn_beta,
            is_cumulative_mode=self.is_cumulative_mode,
            regularization=self.regularization,
            max_width=self.max_width,
            use_leaves_v=self.use_leaves_v,
        )

    @classmethod
    def load(cls, path, device="auto", env=None, lr_schedule=None):
        """
        Load model from path.

        :param path: (str)
        :param device: (Union[th.device, str]) Device on which the policy should be loaded.
        :return: (BasePolicy)
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables["data"], lr_schedule=lr_schedule, step_env=env)  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def predict(self, obs: th.Tensor, state=None, mask=None, deterministic: bool = False):
        return self.forward(th.tensor(obs, dtype=th.float32, device=get_device()), deterministic)[0], None
