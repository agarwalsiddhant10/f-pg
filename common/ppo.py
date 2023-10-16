import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import common.ppo_agent as ppo_agent
from sklearn import neighbors


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, reward_func, state_indices, gamma=0.99, lam=0.95, device=torch.device("cpu"), no_vf=False, **density_kwargs):
        self.obs_buf = np.zeros(ppo_agent.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(ppo_agent.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(ppo_agent.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.dones_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.no_vf = no_vf

        self.reward_func = reward_func
        self.state_indices = state_indices
        self.density_kwargs = density_kwargs
        self.device = device
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, next_obs, rew, val, logp, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.next_obs_buf[self.ptr] = next_obs
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.dones_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0):
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

        path_slice = slice(self.path_start_idx, self.ptr)
        
        self.rew_buf[path_slice] = ((self.rew_buf[path_slice]).reshape(-1, 1) + self.reward_func.get_scalar_reward(self.next_obs_buf[path_slice])).squeeze(-1)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = ppo_agent.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = ppo_agent.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def update_rewards_in_buffer(self, actor=None, last_val=0):
        trajectory_ends = self.path_start_idx + np.where(self.dones_buf[self.path_start_idx:self.ptr] == 1)[0]
        trajectory_ends = np.concatenate((trajectory_ends, [self.ptr-1]))
        start = self.path_start_idx
        for end in trajectory_ends:
            if start > end:
                start = end + 1
                continue
            path_slice = slice(start, end+1)
            self.rew_buf[path_slice] = ((self.rew_buf[path_slice]).reshape(-1, 1) + self.reward_func.get_scalar_reward(self.next_obs_buf[path_slice])).squeeze(-1)

            # append rews based on if last or not
            if end +1 == self.ptr:
                rews = np.append(self.rew_buf[path_slice], last_val)
                vals = np.append(self.val_buf[path_slice], last_val)
            else:
                _, last, _ = actor.step(torch.tensor(self.next_obs_buf[end]).to(self.device))
                rews = np.append(self.rew_buf[path_slice], 0)
                vals = np.append(self.val_buf[path_slice], 0)
            
            # the next line computes rewards-to-go, to be targets for the value function
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = ppo_agent.discount_cumsum(deltas, self.gamma * self.lam)

            self.ret_buf[path_slice] = ppo_agent.discount_cumsum(rews, self.gamma)[:-1]
            
            start = end+1
        self.path_start_idx = self.ptr

    def update_reward_function(self, expert_samples=None):
        # If our reward function
        if expert_samples is not None:
            agent_density = neighbors.KernelDensity(bandwidth=self.density_kwargs['agent']['bandwidth'], kernel=self.density_kwargs['agent']['kernel'])
            agent_density.fit(self.next_obs_buf[self.path_start_idx:self.ptr, self.state_indices])
            # self.agent_kde.fit(self.next_obs_buf[self.path_start_idx:self.ptr, self.state_indices])
            
            expert_density = neighbors.KernelDensity(bandwidth=self.density_kwargs['expert']['bandwidth'], kernel=self.density_kwargs['expert']['kernel'])
            expert_density.fit(expert_samples)

            self.reward_func.update(agent_density, expert_density)

        else:
            self.reward_func.update(self.obs_buf[:self.ptr], self.next_obs_buf[:self.ptr])

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0

    def get_sample(self, batch_size):
        random_ids = np.random.randint(0, high=self.ptr, size=batch_size)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf[random_ids], act=self.act_buf[random_ids], ret=self.ret_buf[random_ids],
                    adv=self.adv_buf[random_ids], logp=self.logp_buf[random_ids])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in data.items()}


class PPO:
    def __init__(self, env, reward_func, state_indices, actor=ppo_agent.MLPActorCritic, seed=0, steps_per_epoch=100000, steps_reward_computation=4000, steps_initial=None, batch_size=256, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                num_pi_updates=80, vf_lr=1e-3, num_v_updates=80, lam=0.97, max_ep_len=1000, target_kl=0.1, device=torch.device('cpu'), no_vf=False, **actor_kwargs):
       

        self.env = env
        self.observation_space = env.observation_space
        # print('observation space: ', self.observation_space)
        self.act_space = env.action_space
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.num_pi_updates = num_pi_updates
        self.num_v_updates = num_v_updates
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.reward_func = reward_func
        self.device=device
        self.state_indices = state_indices
        self.no_vf = no_vf

        if steps_initial is None:
            self.buffer_size = 2 * self.steps_per_epoch + steps_reward_computation
        else:
            # print('Steps initial: ', steps_initial)
            # print('Steps per epoch: ', self.steps_per_epoch)
            self.buffer_size = 10 * (steps_initial + self.steps_per_epoch)

        print('buffer size: ', self.buffer_size)
        self.actor = actor(self.observation_space, self.act_space).to(self.device)
        self.reward_func = reward_func
        self.buffer = PPOBuffer(self.observation_space.shape[0], self.act_space.shape[0], self.buffer_size, self.reward_func, state_indices=state_indices, gamma=self.gamma, device = self.device, no_vf=no_vf, **actor_kwargs['density'])

        self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.actor.v.parameters(), lr=self.vf_lr)

    def compute_loss_pi(self, data):
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']

        # Policy loss
        pi, logp = self.actor.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        if self.no_vf:
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * ret
            loss_pi = -(torch.min(ratio * ret, clip_adv)).mean()
        else:
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.actor.v(obs) - ret)**2).mean()

    def update(self):
        avg_pi_loss = 0.0
        avg_v_loss = 0.0
        num_pi_upd = 0
        avg_kl = 0
        for i in range(self.num_pi_updates):
            data = self.buffer.get_sample(self.batch_size)
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            
            if kl > 1.5 * self.target_kl and num_upd > 0:
                print('Early stopping at step %d due to reaching max kl.'%i)
            loss_pi.backward()
            self.pi_optimizer.step()
            avg_kl += kl
            avg_pi_loss += loss_pi.item()
            num_pi_upd +=1 

        avg_pi_loss /= num_pi_upd
        avg_kl /= num_pi_upd
        
        if not self.no_vf:
            for i in range(self.num_v_updates):
                data = self.buffer.get_sample(self.batch_size)
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(data)
                loss_v.backward()
                self.vf_optimizer.step()
                avg_v_loss += loss_v.item()

        avg_v_loss /= self.num_v_updates


        return avg_pi_loss, avg_kl, avg_v_loss

    def collect_data(self, num_steps, reward_computation=True):
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for t in range(num_steps):
            a, v, logp = self.actor.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))

            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t==num_steps-1

            self.buffer.store(np.copy(o), np.copy(a), np.copy(next_o), np.copy(r), np.copy(v), np.copy(logp), np.copy(terminal))
            
            # Update obs (critical!)
            o = next_o

            if terminal or epoch_ended:
                # if epoch_ended and not(terminal):
                    # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                
                if timeout or epoch_ended:
                    _, v, _ = self.actor.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))
                else:
                    v = 0

                if reward_computation:
                    self.buffer.finish_path(v)

                o, ep_ret, ep_len = self.env.reset(), 0, 0

    def collect_data_pointmaze(self, num_steps, num_steps_reward_computation):
        print('Collecting data for point maze')
        self.actor.cpu()
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            num_steps_collected_goal = 0
            o = self.env.reset()
            initial_state = np.copy([o[0], o[1]])
            ep_ret, ep_len = 0, 0
            goal = self.env.env.n_goal
            num_episodes =0
            while num_steps_collected_goal < num_steps_reward_computation:
                a, v, logp = self.actor.step(torch.as_tensor(o, dtype=torch.float32))
                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                num_steps_collected_goal += 1
                # save and log
                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                self.buffer.store(np.copy(o), np.copy(a), np.copy(next_o), np.copy(r), np.copy(v), np.copy(logp), np.copy(terminal))
                # Update obs (critical!)
                o = next_o
                if terminal:
                    ep_len = 0
                    num_episodes += 1

                    if num_episodes > 10:
                        break
                    o = self.env.reset(False, init=initial_state)
                    
            self.buffer.update_reward_function(expert_samples=goal.reshape(1,-1))
            self.buffer.update_rewards_in_buffer(actor=self.actor)
            num_steps_collected += num_steps_collected_goal

        self.actor.to(self.device)


    def get_action(self, obs, deterministic=False):
        obs = obs[None, :]
        obs = torch.FloatTensor(obs).to(self.device)
        return self.actor.act(obs, deterministic)