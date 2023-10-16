import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import gym
from ruamel.yaml import YAML

from common.pg import PGBuffer, PG
from fil.reward import Reward

import envs
from envs.tasks.grid_task import expert_density
from utils import system, collect, logger, eval
from utils.plots.train_plot import plot, plot_disc, plot_submission
from sklearn import neighbors 

import datetime
import dateutil.tz
import json, copy
torch.set_printoptions(edgeitems=3)

def try_evaluate(itr: int, policy_type: str, old_reward=None, reward=None, agent_density=None, rho_expert=None):
    update_time = itr
    env_steps =  itr * v['pg']['steps_reward_computation'] + v['pg']['num_steps_per_collect']

    agent_emp_states = policy.buffer.next_obs_buf[:, state_indices].copy()

    print(expert_samples.shape)
    print(agent_emp_states.shape)
    metrics = eval.KL_summary(expert_samples, agent_emp_states, 
                         env_steps, policy_type, task_name == 'gaussian')

    plot_submission(agent_emp_states.reshape(-1, v['env']['T'], agent_emp_states.shape[1]), reward.get_scalar_reward, v['obj'],
            log_folder, env_steps, range_lim, metrics, rho_expert, agent_density)
    
    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name, task_name = v['env']['env_name'], v['task']['task_name']
    add_time, state_indices = v['env']['add_time'], v['env']['state_indices']
    seed = v['seed']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    print('running on device: ', device)
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()

    # logs
    exp_id = f"logs/{v['dir']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    log_folder = exp_id
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    print('pid', pid)
    if not os.path.exists(os.path.join(log_folder, 'plt')):
        os.makedirs(os.path.join(log_folder, 'plt'))
    if v['save_interval'] > 0 and not os.path.exists(os.path.join(log_folder, 'model')):
        os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name, **v['env'])
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    # rho_expert and samples for KL estimation (not training)
    rho_expert = expert_density(**v['task'], env=gym_env)
    range_lim = collect.get_range_lim(env_name, v['task'], gym_env)

    if task_name in ['gaussian', 'mix_gaussian']:
        expert_samples = collect.gaussian_samples(env_name, v['task'], gym_env, range_lim)
    elif 'uniform' in task_name:
        expert_samples = collect.expert_samples(env_name, v['task'], rho_expert, range_lim)

    # Initilialize reward as a neural network

    reward = Reward(state_indices, v['env']['T'], obj=v['obj'])

    policy = PG(gym_env, reward_func=reward, state_indices=state_indices, seed=seed, device=device, **v['pg'])
    # policy_optimizer = torch.optim.Adam(policy.parameters(), lr=v['pg']['lr'], weight_decay=v['pg']['weight_decay'], betas=(v['pg']['momentum'], 0.999))

    for itr in range(v['n_itrs']):
        policy.buffer.reset()
        num_steps_per_collect = v['pg']['num_steps_per_collect']
        num_collections = v['pg']['steps_per_epoch'] // num_steps_per_collect

        policy.collect_data(v['pg']['steps_reward_computation'], reward_computation=False)
        policy.buffer.update_reward_function(expert_density=rho_expert)
        policy.buffer.update_rewards_in_buffer()
        # print(policy.buffer.ptr)

        mean_policy_loss = 0
        mean_kl = 0
        for pg_step in range(num_collections):
            policy.collect_data(num_steps_per_collect)
            policy_loss, kl, _ = policy.update()

            mean_policy_loss += policy_loss
            mean_kl += kl

        mean_policy_loss /= num_collections
        mean_kl /= num_collections
        
        # evaluating the learned reward
        try_evaluate(itr, "Running", reward=policy.buffer.reward_func, agent_density=policy.buffer.reward_func.agent_density, rho_expert=rho_expert)

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Policy loss", mean_policy_loss)
        logger.record_tabular("KL", mean_kl)
        
        if v['save_interval'] > 0 and (itr % v['save_interval'] == 0 or itr == v['n_itrs']-1):
            torch.save(policy.actor.state_dict(), os.path.join(logger.get_dir(), f"model/policy_{itr}.pkl"))

        logger.dump_tabular()