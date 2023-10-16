import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    # layers.append(nn.LayerNorm(sizes[0]))
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        # if act is not None:
        #     print(obs)
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, var_net=True):
        super().__init__()
        self.var_net = var_net

        if not self.var_net:
            log_std =  -0.5 * np.ones(act_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
            self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

        else:
            # self.base = mlp([obs_dim] + list(hidden_sizes), activation, activation)
            self.log_std_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
            self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation=nn.Tanh)

    def _distribution(self, obs):
        if not self.var_net:
            std = torch.exp(self.log_std)
            mu = self.mu_net(obs)

        else:
            # base = self.base(obs)
            std = torch.exp(torch.clip(self.log_std_net(obs), -5.0, 5.0))
            mu = self.mu_net(obs)
            
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            if deterministic:
                a = pi.mean
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs, deterministic=False):
        res = self.step(obs, deterministic)
        return res[0], res[2]

class MLPActor(nn.Module):
    def __init__(self, observation_space, action_space, 
                hidden_sizes=(64, 64), activation=nn.Tanh, var_net=True, add_exploration_noise=False, rand_exp_prob=0.0):

        super().__init__()

        obs_dim = observation_space.shape[0]

        self.add_exploration_noise = add_exploration_noise
        self.rand_exp_prob = rand_exp_prob
        # self.noise_std = noise_std
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, var_net=var_net)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

    def step(self, obs, deterministic=False):
        # print(obs.shape)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            if deterministic:
                # print('Deterministic')
                a = pi.mean
                a = torch.clip(a, min=-1.0, max=1.0)
                logp_a = self.pi._log_prob_from_distribution(pi, a)
            else:
                if self.add_exploration_noise:
                    rand_exp = torch.distributions.Uniform(torch.ones_like(pi.mean).to(obs.device) * -1.0, torch.ones_like(pi.mean).to(obs.device) * 1.0)
                    rand = torch.rand(1)
                    if rand < self.rand_exp_prob:
                        a = rand_exp.sample()
                        logp_a = rand_exp.log_prob(a).sum(-1)
                    else:
                        # exp_pi = torch.distributions.Normal(pi.mean, pi.stddev + 0.5)
                        a = pi.sample()
                        a = torch.clip(a, min=-1.0, max=1.0)
                        # a = torch.clip(a, min=-1.0, max=1.0)
                        # logp_a = exp_pi.log_prob(a).sum(-1)
                        logp_a = self.pi._log_prob_from_distribution(pi, a)
                        # print(logp_a)
                    logp_a = torch.log(self.rand_exp_prob * torch.exp(rand_exp.log_prob(torch.clip(a, min=-1.0, max=1.0)).sum(-1)) + (1-self.rand_exp_prob) *torch.exp( self.pi._log_prob_from_distribution(pi, a)))
                else:
                    a = pi.sample()
                    logp_a = self.pi._log_prob_from_distribution(pi, a)
                    # print(logp_a)

        return a.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs, deterministic=False):
        return self.step(obs, deterministic)

    def log_prob(self, obs, act):
        pi, logp_a = self.pi(obs, act)
        if self.add_exploration_noise:
            rand_exp = torch.distributions.Uniform(torch.ones_like(pi.mean).to(obs.device) * -1.0, torch.ones_like(pi.mean).to(obs.device) * 1.0)
            logp_a = torch.log(self.rand_exp_prob * torch.exp(rand_exp.log_prob(torch.clip(act, min=-1.0, max=1.0)).sum(-1)) + (1 - self.rand_exp_prob) * torch.exp(logp_a))

        return pi, logp_a
    
    def pi_correction(self, logp, act):
        rand_exp = torch.distributions.Uniform(torch.ones_like(act).to(act.device) * -1.0, torch.ones_like(act).to(act.device) * 1.0)
        return torch.log(self.rand_exp_prob * torch.exp(rand_exp.log_prob(torch.clip(act, min=-1.0, max=1.0)).sum(-1)) + (1-self.rand_exp_prob) *torch.exp(logp))