import hydra
from omegaconf import DictConfig, OmegaConf
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
from utils.gcwrapper import GCWrapper
import datetime
import dateutil.tz
import json, copy
import random
torch.set_printoptions(edgeitems=3)


def try_evaluate(v, gym_env, policy, itr: int, policy_type: str):
    update_time = itr
    env_steps =  (itr+1) * (v['pg']['steps_initial'] + v['pg']['steps_per_epoch'])

    real_return_det, avg_ep_len, avg_score = eval.evaluate_real_return(policy, gym_env,
                                            v['eval_episodes'], v['env']['T'], True)

    real_return_st, avg_ep_len_st, avg_score_st = eval.evaluate_real_return(policy, gym_env,
                                            v['eval_episodes'], v['env']['T'], True)

    logger.record_tabular("Real Det Return", round(real_return_det, 2))
    logger.record_tabular("Real Det Ep Len", round(avg_ep_len, 2))
    logger.record_tabular("Real Det Score", round(avg_score, 2))

    logger.record_tabular("Real Sto Return", round(real_return_st, 2))
    logger.record_tabular("Real Sto Ep Len", round(avg_ep_len_st, 2))
    logger.record_tabular("Real Sto Score", round(avg_score_st, 2))
    
    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

@hydra.main(config_path="../configs/", config_name="pg_fetch.yaml")
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
    env_fn = lambda: GCWrapper(gym.make(env_name), max_steps=v['env']['T'])
    gym_env = env_fn()
    gym_env.env.seed(seed)
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    # Initilialize reward as a neural network

    reward = Reward(gym_env.goal_state_indices, v['env']['T'], obj=v['obj'])

    policy = PG(gym_env, reward_func=reward, state_indices=gym_env.goal_state_indices, seed=seed, device=device, **v['pg'])
    
    for itr in range(v['n_itrs']):
        policy.buffer.reset()
        num_steps_per_collect = v['pg']['num_steps_per_collect']
        num_collections = v['pg']['steps_per_epoch'] // num_steps_per_collect
        # print(v['pg'])
        print('num_steps_per_collect', num_steps_per_collect)
        print('num_collections', num_collections)

        if 'PointMaze' in env_name:
            policy.collect_data_pointmaze(num_steps_per_collect, v['pg']['steps_reward_computation'])
        else:
            policy.collect_data_goal_conditioned(v['pg']['steps_initial'], v['pg']['steps_reward_computation'])
        # policy.collect_data_goal_conditioned_her(100, 50)
        # policy.visualize_buffer()
        # break
        mean_policy_loss = 0
        mean_kl = 0
        mean_ent = 0
        for pg_step in range(num_collections):
            if 'PointMaze' in env_name:
                policy.collect_data_pointmaze(num_steps_per_collect, v['pg']['steps_reward_computation'])
            else:
                policy.collect_data_goal_conditioned(num_steps_per_collect, v['pg']['steps_reward_computation'])
            policy_loss, kl, ent = policy.update()

            mean_policy_loss += policy_loss
            mean_kl += kl
            mean_ent += ent

        mean_policy_loss /= num_collections
        mean_kl /= num_collections
        mean_ent /= num_collections
        
        # evaluating the learned reward
        try_evaluate(v, gym_env, policy, itr, "Running")

        logger.record_tabular("Iteration", itr)
        logger.record_tabular("Policy loss", mean_policy_loss)
        logger.record_tabular("KL", mean_kl)
        logger.record_tabular("Entropy", mean_ent)
        
        if v['save_interval'] > 0 and (itr % v['save_interval'] == 0 or itr == v['n_itrs']-1):
            torch.save(policy.actor.state_dict(), os.path.join(logger.get_dir(), f"model/policy_{itr}.pkl"))

        logger.dump_tabular()

if __name__ == "__main__":
    main()
