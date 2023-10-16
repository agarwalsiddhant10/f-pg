"""
GAIL file
"""
import numpy as np
import torch
from torch import nn
# from torch.nn import utils
import torch.nn.functional as f
import random
from policy import MlpNetwork, SoftQLearning
from grid_mdp import GridMDP, MazeWorld, WindyMazeWorld, ToroidWorld
import matplotlib.pyplot as plt
from buffers import ReplayBuffer, PGReplayBuffer
import argparse
import os
from os import path
from actor import DiscreteActor

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='random seed', type=int, default=1123)
parser.add_argument('--obj', help="f-dive to use ['fkl', 'rkl']", type=str,
                    default='fkl')
parser.add_argument('--dir', help="directory to save results in", type=str,
                    default='aim_results')
parser.add_argument('--disc', action='store_true')
args = parser.parse_args()
torch.set_default_dtype(torch.float32)
# Set random seeds
seed = 42 * args.seed
print(args.seed)
torch.manual_seed(seed)
random.seed = seed
np.random.seed = seed


class PG:
    """
    Class to take the continuous MDP and use gail to match given target distribution
    """

    def __init__(self, args):
        self.env = MazeWorld()
        # self.env = ToroidWorld()
        self.policy = DiscreteActor(input_dim=self.env.dims, action_space=len(self.env.action_space), max_state=self.env.max_state, min_state=self.env.min_state)
        self.discount = 0.99
        self.check_state = set()
        self.agent_buffer = PGReplayBuffer(size=5000)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-3)  # , lr=3e-4)
        self.args = args

        self.set_target_distribution()

    def reward_function(self, states):
        p_e = self.target_distribution[states[:, 0].cpu().numpy().astype(np.int), states[:, 1].cpu().numpy().astype(np.int)]
        p_theta = self.p_theta[states[:, 0].cpu().numpy().astype(np.int), states[:, 1].cpu().numpy().astype(np.int)]
        if args.obj == 'fkl':
            return np.log(p_e/p_theta)
        if args.obj == 'rkl':
            return p_e/p_theta

    def gather_data(self, num_trans=100) -> None:
        """
        Gather data from current policy
        used to:
        * fit value function
        * update policy
        * plot histograms
        :param num_trans:
        :return:
        """
        t = 0
        while t < num_trans:
            s = self.env.reset()
            s = torch.tensor(s).type(torch.float32).reshape([-1, self.env.dims])
            done = False
            # self.agent_buffer.new_episode()
            env_t = 0
            running_return = 1.0
            while not done:
                # self.states.append(deepcopy(s))
                action, log_a = self.policy.sample_action(s)
                # self.actions.append(a)
                a = np.squeeze(action.data.detach().numpy())
                s_p, r, done, _ = self.env.step(a)
                s_p = torch.tensor(s_p).type(torch.float32).reshape([-1, self.env.dims])
                # d = self.discriminator(sp)
                # i_r = gail_reward(d)
                # self.next_states.append(deepcopy(s))
                # self.rewards.append(i_r)  # deepcopy(r))
                # self.dones.append(deepcopy(done))
                self.agent_buffer.add(s.squeeze(), action.reshape([-1]).detach(), r, s_p.squeeze(), done, running_return)
                # if s_p not in self.check_state:
                #     self.check_state.add(s_p)
                #     self.target_buffer.add(s, a, r, s_p, done)
                s = s_p
                t += 1
                env_t += 1
                if args.disc:
                    running_return *= self.discount
                # if done:
                #     self.agent_buffer.update_returns(self.discount)
            # self.states.append(s)
        self.replay_buffer_dist()
        self.agent_buffer.update_returns(self.reward_function, self.discount)

    def generate_state_distribution(self, num_trajectories=100):
        traj_id = 0
        states = []
        num_trans = 0
        while traj_id < num_trajectories:
            done = False
            s = self.env.reset()
            # if s[0] == self.env.target_state[0][0] and s[1] == self.env.target_state[0][1]:
                # print('Target state reset: ', s)
            states.append(s)
            while not done:
                s = torch.tensor(s).type(torch.float32).reshape([-1, self.env.dims])
                a, log_a = self.policy.sample_action(s)
                a = np.squeeze(a.data.detach().numpy())
                s_p, r, done, _ = self.env.step(a)
                # if s_p[0] == self.env.target_state[0][0] and s_p[1] == self.env.target_state[0][1]:
                # print('Target state: ', s_p)
                # print('Prev state: ', s)
                # print('Done: ', done)
                states.append(np.copy(s_p))
                num_trans += 1
                s = np.copy(s_p)
            traj_id += 1

        all_states = np.zeros((10, 10))
        for state in states:
            # if state[0] == self.env.target_state[0][0] and state[1] == self.env.target_state[0][1]:
            # print('Target state: ', state)
            all_states[int(state[0]), int(state[1])] += 1
        all_states /= num_trans
        return all_states

    def zero_correction(self, prob_dist):
        prob_dist = np.copy(prob_dist)
        eps = 1e-3
        prob_dist += eps
        prob_dist/=np.sum(prob_dist)
        return prob_dist

    def replay_buffer_dist(self, num_samples=5000):
        _, _, _, next_states, _, weights = self.agent_buffer.sample(num_samples)
        agent_vals, agent_counts = np.unique(next_states, axis=0, return_counts=True)

        prob_dist = np.zeros((10, 10))
       
        if args.disc:
            for state, w in zip(next_states, weights):
                prob_dist[int(state[0]), int(state[1])] += w
        else:
            prob_dist[agent_vals[:, 0].astype(np.int), agent_vals[:, 1].astype(np.int)] = agent_counts

        self.p_theta = prob_dist/num_samples

        self.p_theta = self.zero_correction(self.p_theta)

        # return prob_dist/num_samples

        # print(prob_dist)

    def set_target_distribution(self):
        target_distribution = self.env.target_distribution()
        self.target_distribution = self.zero_correction(target_distribution)
        print(self.target_distribution)


    def optimize_policy(self):
        num_samples=5000
        self.policy_optimizer.zero_grad()
        states, actions, returns, _, _, _ = self.agent_buffer.sample(num_samples)
        loss = self.policy.loss(states, actions, returns)
        loss.backward()
        self.policy_optimizer.step()


    def plot_dist(self, num_samples=100, it=0, dname='aim'):
        """
        plot the two distributions as histograms
        :return:
        """
        # dname = 'r_neg'
        if not path.exists(dname):
            os.mkdir(dname)

        # _, _, _, target_distribution, _ = self.target_buffer.sample(num_samples)
        states, _, _, next_states, _, _ = self.agent_buffer.sample(num_samples)
        target_dist = np.reshape(self.env.target_distribution(), (-1,))
        target_distribution = np.random.choice(target_dist.shape[0], num_samples, p=target_dist)
        target_distribution = target_distribution.reshape([-1, 1]).astype(np.float32)
        if self.env.dims > 1:
            target_distribution = np.concatenate([target_distribution, target_distribution], axis=-1)
            target_distribution[:, 0] = target_distribution[:, 0] // self.env.y_dim
            target_distribution[:, 1] = target_distribution[:, 1] % self.env.y_dim
        # target_distribution += np.random.normal(loc=0, scale=0.5, size=target_distribution.shape)
        next_states = next_states.numpy().reshape([-1, self.env.dims]).astype(np.float32)
        # next_states += np.random.normal(loc=0., scale=0.01, size=next_states.shape)
        # q, v, qt, vt = self.policy(states)
        # print(f"q: {np.mean(q.detach().numpy())}, v: {np.mean(v.detach().numpy())},"
        #       f" qt: {np.mean(qt.detach().numpy())}, vt: {np.mean(vt.detach().numpy())}")
        if self.env.dims == 1:
            xloc = np.arange(0, self.env.num_states)
            target_distribution = to_one_hot(target_distribution, self.env.num_states).numpy()
            plt.bar(xloc, np.sum(target_distribution, axis=0), color='r', alpha=0.3, label='target')
            next_states = to_one_hot(next_states, self.env.num_states).numpy()
            plt.bar(xloc, np.sum(next_states, axis=0), color='b', alpha=0.3, label='agent')
            for t in self.env.target_state:
                plt.axvline(x=t, color='r', linestyle='dashed', linewidth=2)
            # sns.kdeplot(np.squeeze(target_distribution), shade=True, color='r', shade_lowest=False, alpha=0.3,
            #             label='target')
            # sns.kdeplot(np.squeeze(next_states), shade=True, color='b', shade_lowest=False, alpha=0.3,
            #             label='agent')
        else:
            from matplotlib.ticker import AutoMinorLocator
            target_vals, target_counts = np.unique(target_distribution, axis=0, return_counts=True)
            agent_vals, agent_counts = np.unique(next_states, axis=0, return_counts=True)
            target_counts = target_counts.astype(np.float) / np.max(target_counts)
            agent_counts = agent_counts.astype(np.float) / np.max(agent_counts)
            # for it in range(target_counts.shape[0]):
            #     plt.plot(target_vals[it, 0] + 0.5, target_vals[it, 1] + 0.5, marker_size=40 * target_counts[it],
            #              color='r', alpha=0.2)
            # for ia in range(agent_counts.shape[0]):
            #     plt.plot(agent_vals[ia, 0] + 0.5, agent_vals[ia, 1] + 0.5, marker_size=40 * agent_counts[ia],
            #              color='b', alpha=0.2)

            plt.xlim(left=0., right=self.env.x_dim)
            plt.ylim(bottom=0., top=self.env.y_dim)
            plt.scatter(target_vals[:, 0] + 0.5, target_vals[:, 1] + 0.5, 200 * target_counts,
                        color='r', alpha=0.5, label='target')
            plt.scatter(agent_vals[:, 0] + 0.5, agent_vals[:, 1] + 0.5, 200 * agent_counts,
                        color='b', alpha=0.5, label='agent')
            plt.xticks(np.arange(self.env.x_dim) + 0.5, np.arange(self.env.x_dim))
            plt.yticks(np.arange(self.env.y_dim) + 0.5, np.arange(self.env.y_dim))
            minor_locator = AutoMinorLocator(2)
            plt.gca().xaxis.set_minor_locator(minor_locator)
            plt.gca().yaxis.set_minor_locator(minor_locator)
            plt.gca().set_aspect('equal')
            plt.grid(which='minor')
            # sns.kdeplot(target_distribution[:, 0], target_distribution[:, 1],
            # shade=True, color='r', shade_lowest=False,
            #             alpha=0.5, label='target')
            # sns.kdeplot(next_states[:, 0], next_states[:, 1], shade=True, color='b', shade_lowest=False, alpha=0.5,
            #             label='agent')
        plt.legend()
        # plt.hist(target_distribution, bins=10, alpha=0.4, color='red')
        # plt.hist(next_states, bins=10, alpha=0.4, color='blue')
        # plt.axvline(x=self.env.target_state, color='r', linestyle='dashed', linewidth=2)
        # plt.legend(['target', 'agent'])
        plt.title(f'Density for agent and target distributions state Iteration {it}')
        # plt.show()
        plt.tight_layout()
        plt.savefig(f'{dname}/d_{it}.png', dpi=300)
        # exit()
        plt.cla()
        plt.clf()

        # reward_func = reward_dict[reward_to_use]
        reward_func = self.reward_function
        if reward_func is not None:
            r_states = []
            for ia in range(self.env.x_dim):
                for ja in range(self.env.y_dim):
                    r_states.append([ia, ja])
            r_states = np.asarray(r_states)
            r_states = torch.tensor(r_states)
            d = reward_func(r_states)
            print(f'Max potential: {np.max(d)}, Min potential: {np.min(d)}')
            rewards = d
            rewards = np.reshape(rewards, newshape=(self.env.x_dim, self.env.y_dim))
            rewards = np.transpose(rewards)
            plt.imshow(rewards, cmap='magma', origin='lower')
            plt.colorbar()
            plt.title(f'Rewards at Iteration {it}')
            plt.tight_layout()
            plt.savefig(f'{dname}/r_{it}.png', dpi=300)
            plt.cla()
            plt.clf()
        entropy_kl = self.policy.entropy(states.reshape([-1, self.env.dims]))
        entropy_kl = np.mean(entropy_kl.detach().numpy())
        print(f"Entropy KL at Iteration {it} is {entropy_kl}")

        if self.env.dims == 1:
            states = np.arange(0, self.env.num_states)
            s = torch.tensor(states).type(torch.float32).reshape([-1, 1])
            # s = to_one_hot(s, self.env.num_states)
            # d = self.discriminator(s)
            reward_func = reward_dict[reward_to_use]
            if reward_func is not None:
                rewards = reward_func(s).squeeze().detach().numpy()
                plt.cla()
                plt.bar(states, rewards, width=0.5)
                plt.xlabel('states')
                plt.ylabel('rewards')
                plt.title('Rewards for entering state')
                plt.show()
                logits = self.policy(s)[0]
                policy = torch.exp(logits).detach().numpy()
                plt.bar(states, policy[:, 0], width=0.5)
                plt.xlabel('states')
                plt.ylabel('P(left|state)')
                plt.title('Policy')
                plt.show()
            plt.cla()


if __name__ == '__main__':
    pg = PG(args)
    pg.gather_data(num_trans=500)
    pg.replay_buffer_dist()
    pg.plot_dist(num_samples=500, dname=args.dir)
     
    print('')
    for i in range(500):
        pg.gather_data(5000)
        # break
        pg.optimize_policy()
        if (i + 1) % 1 == 0:
            # gather more data if you want to see exactly what the agent's policy is
            # pg.gather_data()
            print(i + 1)
            print('Plotting')
            pg.plot_dist(num_samples=500, it=(i + 1), dname=args.dir)
