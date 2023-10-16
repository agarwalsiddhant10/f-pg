import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import common.ppo_agent as ppo_agent
from sklearn import neighbors
from common.kde import GaussianKernel
import time

class PGBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, reward_func, state_indices, gamma=0.99, device=torch.device('cpu'), **density_kwargs):
        self.obs_buf = np.zeros(ppo_agent.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(ppo_agent.combined_shape(size, act_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(ppo_agent.combined_shape(size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.prev_rew_buf = np.zeros(size, dtype=np.float32)
        self.dones_buf = np.zeros(size, dtype=np.bool)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.weights_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.density_kwargs = density_kwargs
        self.reward_func = reward_func
        self.state_indices = state_indices
        self.device = device

        # self.agent_kde = GaussianKernel(self.density_kwargs['agent']['bandwidth'])
        # self.expert_kde = GaussianKernel(self.density_kwargs['expert']['bandwidth'])

    def store(self, obs, act, next_obs, logp, done, weight=1.0):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.next_obs_buf[self.ptr] = next_obs
        # self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.dones_buf[self.ptr] = done
        self.weights_buf[self.ptr] = weight
        self.ptr += 1
        # print(self.ptr, done, self.dones_buf[self.ptr-1])

    def store_trajectories(self, obs, act, next_obs, logp, done):
        """
        Append one trajectory of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr:self.ptr+len(obs)] = obs
        self.act_buf[self.ptr:self.ptr+len(obs)] = act
        self.next_obs_buf[self.ptr:self.ptr+len(obs)] = next_obs
        # self.rew_buf[self.ptr:self.ptr+len(obs)] = rew
        self.logp_buf[self.ptr:self.ptr+len(obs)] = logp
        self.dones_buf[self.ptr:self.ptr+len(obs)] = done
        self.ptr += len(obs)
        # print(self.ptr, done, self.dones_buf[self.ptr-1])
        
    def finish_path(self):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # print(self.path_start_idx, self.ptr)
        path_slice = slice(self.path_start_idx, self.ptr)
        self.rew_buf[path_slice] = self.reward_func.get_scalar_reward(self.next_obs_buf[path_slice])
        self.prev_rew_buf[path_slice] = self.reward_func.get_scalar_reward(self.obs_buf[path_slice])
        # self.rew_buf[path_slace] = self.gamma * self.reward_func.get_scalar_reward(self.next_obs_buf[path_slice]) - self.reward_func.get_scalar_reward(self.obs_buf[path_slice])
        rews = np.append(self.rew_buf[path_slice], 0)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = ppo_agent.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def update_rewards_in_buffer(self):
        trajectory_ends = self.path_start_idx + np.where(self.dones_buf[self.path_start_idx:self.ptr] == 1)[0]
        trajectory_ends = np.concatenate((trajectory_ends, [self.ptr-1]))
        start = self.path_start_idx
        for end in trajectory_ends:
            if start > end:
                start = end + 1
                continue
            path_slice = slice(start, end+1)
            self.rew_buf[path_slice] = self.reward_func.get_scalar_reward(self.next_obs_buf[path_slice])
            # self.rew_buf[path_slice] = self.gamma * self.reward_func.get_scalar_reward(self.next_obs_buf[path_slice]) - self.reward_func.get_scalar_reward(self.obs_buf[path_slice])
            rews = np.append(self.rew_buf[path_slice], 0)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = ppo_agent.discount_cumsum(rews, self.gamma)[:-1]
            
            start = end+1
        self.path_start_idx = self.ptr


    def update_reward_function(self, expert_samples=None, expert_density=None):
        assert not (expert_density is None and expert_samples is None)

        # agent_density = neighbors.KernelDensity(bandwidth=self.density_kwargs['agent']['bandwidth'], kernel=self.density_kwargs['agent']['kernel'])
        # agent_density.fit(self.next_obs_buf[self.path_start_idx:self.ptr, self.state_indices])
        agent_density = GaussianKernel(self.density_kwargs['agent']['bandwidth'])
        agent_density.fit(self.next_obs_buf[self.path_start_idx:self.ptr, self.state_indices], self.weights_buf[self.path_start_idx:self.ptr])

        if expert_density is not None:
            self.reward_func.update(agent_density, expert_density)

        else:
            # expert_density = neighbors.KernelDensity(bandwidth=self.density_kwargs['expert']['bandwidth'], kernel=self.density_kwargs['expert']['kernel'])
            expert_density = GaussianKernel(self.density_kwargs['expert']['bandwidth'])
            expert_density.fit(expert_samples)

            self.reward_func.update(agent_density, expert_density)

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def get_sample(self, batch_size):
        random_ids = np.random.randint(0, high=self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[random_ids], act=self.act_buf[random_ids], ret=self.ret_buf[random_ids], logp=self.logp_buf[random_ids], rew=self.rew_buf[random_ids], prev_rew=self.prev_rew_buf[random_ids])
        return {k:torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in data.items()}

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0




class PG:
    def __init__(self, env, reward_func, state_indices, actor=ppo_agent.MLPActor, steps_per_epoch=100000, steps_reward_computation=4000, steps_initial=None, batch_size=256, gamma=0.99, clip_ratio=0.2, lr=3e-4,
                num_pi_updates=80, max_ep_len=1000, target_kl=0.1, device=torch.device('cpu'), action_reg = 0.0, **actor_kwargs):
       

        self.env = env
        self.observation_space = env.observation_space
        # print('observation space: ', self.observation_space)
        self.act_space = env.action_space
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.num_updates = num_pi_updates
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.reward_func = reward_func
        self.device=device
        self.state_indices = state_indices
        self.action_reg = action_reg

        if steps_initial is None:
            self.buffer_size = 2 * self.steps_per_epoch + steps_reward_computation
        else:
            # print('Steps initial: ', steps_initial)
            # print('Steps per epoch: ', self.steps_per_epoch)
            self.buffer_size = 10 * (steps_initial + self.steps_per_epoch)

        print('buffer size: ', self.buffer_size)
        self.actor = actor(self.observation_space, self.act_space, var_net=actor_kwargs['var_net']).to(self.device)
        self.reward_func = reward_func
        self.buffer = PGBuffer(self.observation_space.shape[0], self.act_space.shape[0], self.buffer_size, self.reward_func, state_indices=state_indices, gamma=self.gamma, device = self.device, **actor_kwargs['density'])

        self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.lr)

    def compute_loss_pi(self, data):
        obs, act, ret, logp_old, rew, prev_rew = data['obs'], data['act'], data['ret'], data['logp'], data['rew'], data['prev_rew']
        new_ret = ret - ret.mean(0)
        new_ret = new_ret - self.action_reg * 0.5 * act.norm(p=2, dim=1) ** 2 
        pi, logp = self.actor.pi(obs, act)
        # logp = self.actor.pi_correction(logp, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*new_ret
        loss_pi = -(torch.min(ratio * new_ret, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def update(self):
        avg_loss = 0.0
        num_upd = 0
        avg_kl = 0
        avg_ent = 0
        for i in range(self.num_updates):
            data = self.buffer.get_sample(self.batch_size)
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            ent = pi_info['ent']
            
            if kl > 1.5 * self.target_kl and num_upd > 0:
                print('Early stopping at step %d due to reaching max kl.'%i)
                continue
            loss_pi.backward()
            self.pi_optimizer.step()
            avg_kl += kl
            avg_ent += ent
            avg_loss += loss_pi.item()
            num_upd +=1 

        avg_loss /= num_upd
        avg_kl /= num_upd
        avg_ent /= num_upd

        return avg_loss, avg_kl, avg_ent

    def collect_data(self, num_steps, reward_computation=True):
        print('Collecting data')
        self.actor.cpu()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        weight = self.gamma
        for t in range(num_steps):
            # print(t)
            a, logp = self.actor.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t==num_steps-1

            self.buffer.store(np.copy(o), np.copy(a), np.copy(next_o), np.copy(logp), np.copy(terminal), np.copy(weight))
            weight *= self.gamma
            
            # Update obs (critical!)
            o = next_o

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if reward_computation:
                    self.buffer.finish_path()
                # else:
                #     self.buffer.update_reward_function
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                weight = self.gamma
        # self.buffer.path_start_idx = self.buffer.ptr
        self.actor.to(self.device)

    def collect_data_goal_conditioned(self, num_steps, num_steps_per_goal):
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            num_steps_collected_goal = 0
            o, ep_ret, ep_len = self.env.reset(), 0, 0
            goal = self.env.env.goal
            # print(goal)
            while num_steps_collected_goal < num_steps_per_goal:

                a, logp = self.actor.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                num_steps_collected_goal += 1

                # save and log
                timeout = ep_len == self.max_ep_len
                terminal = d or timeout

                self.buffer.store(np.copy(o), np.copy(a), np.copy(next_o), np.copy(logp), np.copy(terminal))
            
                # Update obs (critical!)
                o = next_o

                if terminal:
                    o = self.env.reset(False)
            
            self.buffer.update_reward_function(expert_samples=goal.reshape(1,-1))
            self.buffer.update_rewards_in_buffer()

            num_steps_collected += num_steps_collected_goal


    def collect_data_pointmaze(self, num_steps, num_steps_reward_computation):
        print('Collecting data for point maze')
        # print('Profiling data collection')
        # avg_time_per_step = 0
        self.actor.cpu()
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            num_steps_collected_goal = 0
            o = self.env.reset()
            initial_state = np.copy([o[0], o[1]])
            ep_ret, ep_len = 0, 0
            goal = self.env.env.goal
            num_episodes =0
            weight = self.gamma
            while num_steps_collected_goal < num_steps_reward_computation:
                # t = time.time()
                a, logp = self.actor.step(torch.as_tensor(o, dtype=torch.float32))
                next_o, r, d, _ = self.env.step(a)
                # avg_time_per_step += time.time() - t
                ep_ret += r
                ep_len += 1
                num_steps_collected_goal += 1
                # save and log
                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                self.buffer.store(np.copy(o), np.copy(a), np.copy(next_o), np.copy(logp), np.copy(terminal), np.copy(weight))
                # Update obs (critical!)
                weight *= self.gamma
                o = next_o
                if terminal:
                    ep_len = 0
                    num_episodes += 1

                    if num_episodes > 10:
                        break
                    o = self.env.reset(False, init=initial_state)
                    weight = self.gamma
                    
            self.buffer.update_reward_function(expert_samples=goal.reshape(1,-1))
            self.buffer.update_rewards_in_buffer()
            num_steps_collected += num_steps_collected_goal

        # avg_time_per_step /= num_steps_collected
        # print('Average time per step: ', avg_time_per_step)

        self.actor.to(self.device)


    def get_action(self, obs, deterministic=False):
        obs = obs[None, :]
        obs = torch.FloatTensor(obs).to(self.device)
        return self.actor.act(obs, deterministic)

    def visualize_buffer(self):
        for i in range(self.buffer.ptr):
            print(i, self.buffer.obs_buf[i][self.state_indices], self.buffer.obs_buf[i][self.state_indices + len(self.state_indices)], self.buffer.rew_buf[i], self.buffer.dones_buf[i], self.buffer.ret_buf[i])