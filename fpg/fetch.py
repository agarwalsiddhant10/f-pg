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
from utils.gcwrapper import GCWrapper
from utils.plots.train_plot import plot, plot_disc, plot_submission
from utils.eval_gc import Eval
from sklearn import neighbors 

import datetime
import dateutil.tz
import json, copy
torch.set_printoptions(edgeitems=3)

def try_evaluate(itr: int, policy_type: str, old_reward=None, reward=None, agent_density=None, rho_expert=None):
    update_time = itr
    env_steps =  itr * (v['pg']['steps_initial'] + v['pg']['steps_per_epoch'])

    # avg_return = evaluator.eval_goal_conditioned_her(1000, 50, itr)

    real_return_det = eval.evaluate_real_return(policy, env_fn(), 
                                            v['eval_episodes'], v['env']['T'], True)

    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    # real_return_sto = eval.evaluate_real_return(policy.get_action, env_fn(), 
    #                                         v['eval_episodes'], v['env']['T'], False)
    
    # logger.record_tabular("Real Sto Return", round(real_return_sto, 2))

    # agent_emp_states = policy.buffer.next_obs_buf[:, state_indices].copy()

    # print(expert_samples.shape)
    # print(agent_emp_states.shape)
    # metrics = eval.KL_summary(expert_samples, agent_emp_states, 
    #                      env_steps, policy_type, task_name == 'gaussian')

    # plot_submission(agent_emp_states.reshape(-1, v['env']['T'], agent_emp_states.shape[1]), reward.get_scalar_reward, v['obj'],
    #         log_folder, env_steps, range_lim, metrics, rho_expert, agent_density)
    
    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
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
    env_fn = lambda: GCWrapper(gym.make(env_name), max_steps=v['env']['T'])
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    # Initilialize reward as a neural network

    reward = Reward(gym_env.goal_state_indices, v['env']['T'], 'fkl')

    policy = PG(gym_env, reward_func=reward, state_indices=gym_env.goal_state_indices, seed=seed, device=device, **v['pg'])
    evaluator = Eval(gym_env, reward_func=reward, state_indices=gym_env.goal_state_indices, actor = policy, output_dir = log_folder, 
        device=device, **v['pg'])
    # policy_optimizer = torch.optim.Adam(policy.parameters(), lr=v['pg']['lr'], weight_decay=v['pg']['weight_decay'], betas=(v['pg']['momentum'], 0.999))

    for itr in range(v['n_itrs']):
        policy.buffer.reset()
        num_steps_per_collect = v['pg']['num_steps_per_collect']
        num_collections = v['pg']['steps_per_epoch'] // num_steps_per_collect

        policy.collect_data_goal_conditioned(v['pg']['steps_initial'], v['pg']['steps_reward_computation'])
        # policy.collect_data_goal_conditioned_her(100, 50)
        # policy.visualize_buffer()
        # break
        mean_policy_loss = 0
        mean_kl = 0
        for pg_step in range(num_collections):
            policy.collect_data_goal_conditioned(num_steps_per_collect, v['pg']['steps_reward_computation'])
            policy_loss, kl, _ = policy.update()

            mean_policy_loss += policy_loss
            mean_kl += kl

        mean_policy_loss /= num_collections
        mean_kl /= num_collections
        
        # evaluating the learned reward
        try_evaluate(itr, "Running")

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Policy loss", mean_policy_loss)
        logger.record_tabular("KL", mean_kl)
        
        if v['save_interval'] > 0 and (itr % v['save_interval'] == 0 or itr == v['n_itrs']-1):
            torch.save(policy.actor.state_dict(), os.path.join(logger.get_dir(), f"model/policy_{itr}.pkl"))

        logger.dump_tabular()