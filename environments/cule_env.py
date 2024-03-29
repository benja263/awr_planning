# Built-in
from collections import deque
import math

# External
import torch
from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom
import gym
import numpy as np
from gym import spaces


class CuleEnv(gym.Env):
    def __init__(self, device, env_kwargs, n_frame_stack=4, noop_max=30, clip_reward=True, fire_reset=True):
        self.device = device
        self.env_kwargs = env_kwargs
        cart = AtariRom(env_kwargs["env_name"])
        actions = cart.minimal_actions()
        print(f"Using {actions} actions")
        self.env = AtariEnv(num_envs=1, device=torch.device("cpu"), **env_kwargs)
        super(AtariEnv, self.env).reset(0)
        self.env.reset(initial_steps=1, verbose=1)
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.n_frame_stack = n_frame_stack  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=n_frame_stack)
        self.training = True  # Consistent with model training mode
        self.noop_max = noop_max
        self.clip_reward = clip_reward
        self.fire_reset = fire_reset

        # Stable baselines requirements
        self.reward_range = (-math.inf, math.inf)
        self.metadata = {"render.modes": ["human", "rgb_array"]}
        orig_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (orig_shape[0] * n_frame_stack, orig_shape[1], orig_shape[2]),
                                                np.uint8)
        self.action_space = spaces.Discrete(len(actions))

    def _reset_buffer(self):
        # print(f"Reset_buffer for {self.n_frame_stack} n_frame_stacks")
        for _ in range(self.n_frame_stack):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        obs = self.reset_no_fire()
        if not self.fire_reset:
            return obs
        obs, _, done, _ = self.step(1)
        if done:
            # print(f"done after step 1")
            self.reset_no_fire()
        obs, _, done, _ = self.step(2)
        if done:
            # print(f"done after step 2")
            self.reset_no_fire()
        return obs

    def reset_no_fire(self):
        # print(f"Reset_no_fire for {self.n_frame_stack} n_frame_stacks")
        obs = torch.zeros(84, 84, device=self.device)
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.env.step(torch.tensor([0]))  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            # Perform up to 30 random no-ops before starting
            noops = np.random.randint(self.noop_max + 1)
            obs = self.env.reset(initial_steps=noops, verbose=1)
            self.env.lives[0] = 5
            obs = obs[0, :, :, 0].to(self.device)
        self.last_frame = obs
        self.state_buffer.append(obs)
        self.lives = self.env.lives.item()
        return torch.stack(list(self.state_buffer), 0).cpu().numpy()

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        obs, reward, done, info = self.env.step(torch.tensor([action]))
        info["orig_reward"] = reward[0].item()
        if self.clip_reward:
            reward = torch.sign(reward)
        if self.lives is None:
            self.lives = self.env.lives.item()
        obs = obs[0, :, :, 0].to(self.device)
        self.state_buffer.append(obs)
        self.last_frame = obs
        # Detect loss of life as terminal in training mode
        lives = info["ale.lives"][0].item()
        if self.training:
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = True  # not done  # Only set flag when not truly done
                done = True
        self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0).cpu().numpy(), reward[0].cpu().numpy(), done, info

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
