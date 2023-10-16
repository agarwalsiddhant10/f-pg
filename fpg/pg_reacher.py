import hydra
from omegaconf import DictConfig, OmegaConf
import random
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import gym
from ruamel.yaml import YAML

from common.pg import PGBuffer, PG
from fpg.reward import Reward

import envs
from envs.tasks.grid_task import expert_density
from utils import logger, eval
from utils.plots.train_plot import plot, plot_submission
from sklearn import neighbors 

import datetime
import dateutil.tz
import json, copy

torch.set_printoptions(edgeitems=3)

def try_evaluate(v, gym_env, policy, reward, rho_expert, log_dir, itr: int, policy_type: str):
    update_time = itr
    env_steps =  itr * (v['pg']['steps_reward_computation']) + v['pg']['num_steps_per_collect']

    agent_emp_states = policy.buffer.next_obs_buf[:policy.buffer.ptr, v['env']['state_indices']].copy()
    print('Agent emp states shape: ', agent_emp_states.shape)

    plot_submission(agent_emp_states, reward.get_scalar_reward, v['obj'],
            log_dir, env_steps, [gym_env.range_x, gym_env.range_y], rho_expert)
    
    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)


@hydra.main(config_path="../configs/", config_name="pg_reacher_trace_gauss.yaml")
def main(v: DictConfig):
    env_name = v['env']['env_name']
    seed = v['seed']
    print(v)

    print("Seed: ", seed)

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    print('running on device: ', device)
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


    pid=os.getpid()

    # logs
    exp_id = f"logs/{v['dir']}/{v['obj']}/{v['seed']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        print('making dir')
        os.makedirs(exp_id)

    log_folder = exp_id
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    # if not os.path.exists(log_folder):
        # os.makedirs(log_folder)
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

    rho_expert = expert_density(**v['task'], env=gym_env)
    state_indices = v['env']['state_indices']
    # range_lim = collect.get_range_lim(env_name, v['task'], gym_env)

    # expert_samples = collect.gaussian_samples(env_name, v['task'], gym_env, range_lim)

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
        mean_ent = 0
        for pg_step in range(num_collections):
            policy.collect_data(num_steps_per_collect)
            policy_loss, kl, ent = policy.update()

            mean_policy_loss += policy_loss
            mean_kl += kl
            mean_ent += ent

        mean_policy_loss /= num_collections
        mean_kl /= num_collections
        mean_ent /= num_collections
        
        # evaluating the learned reward
        try_evaluate(v, gym_env, policy, reward, rho_expert, log_folder, itr, 'PG')

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Policy loss", mean_policy_loss)
        logger.record_tabular("KL", mean_kl)
        logger.record_tabular("Entropy", mean_ent)
        
        if v['save_interval'] > 0 and (itr % v['save_interval'] == 0 or itr == v['n_itrs']-1):
            torch.save(policy.actor.state_dict(), os.path.join(logger.get_dir(), f"model/policy_{itr}.pkl"))

        logger.dump_tabular()

if __name__ == "__main__":
    main()


    


