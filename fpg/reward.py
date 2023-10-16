import torch
import numpy as np
from sklearn import neighbors
from discriminator import Discriminator


class Reward:
    def __init__(self, state_indices, ep_len, obj):
        self.agent_density = None
        self.expert_density = None
        print('State indices: ', state_indices)
        self.state_indices = state_indices
        self.ep_len = ep_len
        self.obj = obj
        print('Objective', self.obj)

    def get_scalar_reward(self, state):
        # print(state.shape)
        # _, d = state.shape
        d = state.shape[-1]
        s_vec = state.reshape(-1, d)
        # print('State shape: ', s_vec.shape)
        if d != len(self.state_indices):
            s_vec = s_vec[:, self.state_indices]
        rho_expert_samples = np.clip(self.expert_density(s_vec), a_min=3e-2, a_max=None)
        agent_density_samples = np.clip(np.exp(self.agent_density(s_vec)), a_min=1e-2, a_max=None)
        reward = rho_expert_samples/agent_density_samples
        if self.obj == 'fkl':
            return np.log(reward)
        elif self.obj == 'rkl':
            return reward
        elif self.obj == 'js':
            return np.log((1 + (1.0/reward))/(2 * (1.0/reward)))
        elif self.obj == 'chi2':
            return -1.0/(reward)

    def update(self, agent_density, expert_density):
        self.agent_density = agent_density
        self.expert_density = expert_density

class L2Reward:
    def __init__(self, state_indices, ep_len, obj):
        print('State indices: ', state_indices)
        self.state_indices = state_indices
        self.ep_len = ep_len
        self.obj = obj

    def get_scalar_reward(self, state):
        return -np.linalg.norm(state[:, self.state_indices] - state[:, self.state_indices + len(self.state_indices)])

    def update(self, obs, next_obs):
        return

class RewardDisc:
    def __init__(self, state_indices, ep_len, obj, scale):
        print('State indices: ', state_indices)
        self.state_indices = state_indices
        self.goal_dim = len(state_indices)
        self.ep_len = ep_len
        self.obj = obj
        self.discriminator = Discriminator(x_dim = 2 * self.goal_dim, reward_type=obj, scale=scale)

    def get_scalar_reward(self, state):
        # print(state.shape)
        return self.discriminator.reward(state[:, -2 * len(self.state_indices):])

    def update(self, obs_data, next_obs_data):
        num_updates = 10
        batch_size = 1024
        self.discriminator.cuda()
        for _ in range(100):
            sample_indices = np.random.choice(obs_data.shape[0], batch_size)
            obs = obs_data[sample_indices, -2 * self.goal_dim:]
            next_obs = next_obs_data[sample_indices, -2 * self.goal_dim:]
            goals = obs_data[sample_indices, -self.goal_dim:]
            noisy_targets = goals + np.random.normal(0, 0.005, size=goals.shape)
            target_obs = np.concatenate([noisy_targets, goals], axis=-1)
            # target_obs = np.clip(target_obs, a_min=-0.0, a_max=1.0)
            # print('Target obs: ', target_obs)
            # print('Policy states: ', obs)
            self.discriminator.optimize_discriminator(target_states=target_obs, policy_states=obs, policy_next_states=next_obs)

class NoneReward:
    def __init__(self, state_indices, ep_len, obj):
        print('State indices: ', state_indices)
        self.state_indices = state_indices
        self.ep_len = ep_len
        self.obj = obj

    def get_scalar_reward(self, state):
        return 0.0

    def update(self, obs, next_obs):
        return
