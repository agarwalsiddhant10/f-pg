import torch
import numpy as np
import gym
import time, copy
import time
import matplotlib.pylab as plt
import os
    
def eval_mujoco(policy, env, n_episodes, dir, itr, deterministic):
    print('Velocity profile...')
    policy.actor.to(torch.device('cpu'))
    policy.device = torch.device('cpu')
    velocities = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.get_action(obs, deterministic)
            obs, rew, done, info = env.step(action.squeeze(0))
            velocities.append(info['velocity'])
    policy.actor.to(torch.device('cuda'))
    policy.device = torch.device('cuda')
    velocities = np.array(velocities)
    dir = os.path.join(dir, 'plt')
    if not os.path.exists(dir):
        os.makedirs(dir)
    plot_name = dir + f'/velocity_profile_{itr}_{deterministic}.png'
    # Plot histogram of velocities
    plt.hist(velocities, bins=20)
    plt.title('Velocity profile')
    plt.xlabel('Velocity')
    plt.ylabel('Count')
    plt.savefig(plot_name)


def evaluate_real_return(policy, env, n_episodes, horizon, deterministic):  
    returns = []
    print('Profiling evaluation...')
    tot_time = time.time()
    avg_time_per_step = 0.0
    num_steps = 0
    avg_ep_len = 0
    avg_score = 0.0
    policy.actor.to(torch.device('cpu'))
    policy.device = torch.device('cpu')
    for _ in range(n_episodes):
        obs = env.reset()
        # print('Achieved goal: ', obs[25:28], ' Desired goal: ', obs[28:31], ' Norm: ', np.linalg.norm(obs[25:28] - obs[28:31], axis=-1))
        ret = 0
        ep_len = 0
        score = 0.0
        for t in range(horizon):
            st = time.time()
            action, _ = policy.get_action(obs, deterministic)
            obs, rew, done, info = env.step(action.squeeze(0)) # NOTE: assume rew=0 after done=True for evaluation
            avg_time_per_step += time.time() - st
            num_steps += 1
            # print(rew)
            ret += rew 
            ep_len += 1
            if rew > 0.0:
                done = True
            if 'reward' in info:
                score += info['reward'] 
            if done:
                break
            
        returns.append(ret)
        # print(returns)
        avg_ep_len += ep_len
        avg_score += score
    policy.actor.to(torch.device('cuda'))
    policy.device = torch.device('cuda')

    avg_time_per_step /= num_steps
            

    print('Eval time per step: {}, number of steps: {}'.format(avg_time_per_step, num_steps))
    print('Total eval time: {}'.format(time.time() - tot_time))

    return np.mean(returns), avg_ep_len / n_episodes, avg_score / n_episodes
    # return np.mean(returns)