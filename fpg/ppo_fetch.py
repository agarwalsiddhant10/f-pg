import hydra
from omegaconf import DictConfig, OmegaConf
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import gym
from ruamel.yaml import YAML

from common.ppo import PPOBuffer, PPO
from fil.reward import Reward, NoneReward, RewardDisc, L2Reward

import envs
from envs.tasks.grid_task import expert_density
from utils import system, collect, logger, eval
from utils.gcwrapper import GCWrapper
from utils.plots.train_plot import plot, plot_disc, plot_submission
from utils.eval_gc import Eval
import datetime
import dateutil.tz
import json, copy
torch.set_printoptions(edgeitems=3)


def try_evaluate(v, env_fn, policy, evaluator, itr: int, policy_type: str):
    update_time = itr
    env_steps =  (itr+1) * (v['ppo']['steps_initial'] + v['ppo']['steps_per_epoch'])

    # if itr % 5 == 0:
    # avg_return = evaluator.eval_disc(4, itr, policy.buffer.reward_func)
    
    real_return_det, avg_ep_len, avg_score = eval.evaluate_real_return(policy, env_fn(), 
                                            v['eval_episodes'], v['env']['T'], True)

    real_return_st, avg_ep_len_st, avg_score_st = eval.evaluate_real_return(policy, env_fn(), 
                                            v['eval_episodes'], v['env']['T'], True)

    logger.record_tabular("Real Det Return", round(real_return_det, 2))
    logger.record_tabular("Real Det Ep Len", round(avg_ep_len, 2))
    logger.record_tabular("Real Det Score", round(avg_score, 2))

    logger.record_tabular("Real Sto Return", round(real_return_st, 2))
    logger.record_tabular("Real Sto Ep Len", round(avg_ep_len_st, 2))
    logger.record_tabular("Real Sto Score", round(avg_score_st, 2))
    
    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

@hydra.main(config_path="/u/siddhant/f-IL/f-PG/configs/samples/agents/", config_name="ppo_fetch.yaml")
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
    system.reproduce(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    # Initilialize reward as a neural network

    if v['obj'] in ['aim', 'gail', 'airl', 'fairl']:
        reward = RewardDisc(gym_env.goal_state_indices, v['env']['T'], obj=v['obj'], scale = 10.0)
    elif v['obj'] == 'l2':
        reward = L2Reward(gym_env.goal_state_indices, v['env']['T'], obj=v['obj'])
    elif v['obj'] == 'none':
        reward = NoneReward(gym_env.goal_state_indices, v['env']['T'], obj=v['obj'])
    else:
        reward = Reward(gym_env.goal_state_indices, v['env']['T'], obj=v['obj'])

    policy = PPO(gym_env, reward_func=reward, state_indices=gym_env.goal_state_indices, seed=seed, device=device, **v['ppo'])
    evaluator = Eval(gym_env, reward_func=reward, state_indices=gym_env.goal_state_indices, actor = policy, output_dir = log_folder, 
        device=device, **v['ppo'])
    # policy_optimizer = torch.optim.Adam(policy.parameters(), lr=v['pg']['lr'], weight_decay=v['pg']['weight_decay'], betas=(v['pg']['momentum'], 0.999))

    for itr in range(v['n_itrs']):
        policy.buffer.reset()
        num_steps_per_collect = v['ppo']['num_steps_per_collect']
        num_collections = v['ppo']['steps_per_epoch'] // num_steps_per_collect
        print(v['ppo'])
        print('num_steps_per_collect', num_steps_per_collect)
        print('num_collections', num_collections)

        if v['obj'] not in ['fkl', 'rkl', 'js', 'chi2']:
            policy.collect_data(v['ppo']['steps_initial'], False)
            policy.buffer.update_reward_function()
            policy.buffer.update_rewards_in_buffer(actor=policy.actor)
        else:
            policy.collect_data_pointmaze(v['ppo']['steps_initial'], v['ppo']['steps_reward_computation'])
        
        # policy.collect_data_goal_conditioned_her(100, 50)
        # policy.visualize_buffer()
        # break
        mean_policy_loss = 0
        mean_value_loss = 0
        mean_kl = 0
        for pg_step in range(num_collections):
            if v['obj'] not in ['fkl', 'rkl', 'js', 'chi2']:
                policy.collect_data(num_steps_per_collect, True)
            else:
                policy.collect_data_pointmaze(num_steps_per_collect, v['ppo']['steps_reward_computation'])

            policy_loss, kl, value_loss = policy.update()

            mean_policy_loss += policy_loss
            mean_kl += kl
            mean_value_loss += value_loss

        mean_policy_loss /= num_collections
        mean_kl /= num_collections
        mean_value_loss /= num_collections
        
        # evaluating the learned reward
        try_evaluate(v, env_fn, policy, evaluator, itr, "Running")

        logger.record_tabular("Iteration", itr)
        logger.record_tabular("Policy loss", mean_policy_loss)
        logger.record_tabular("KL", mean_kl)
        
        if v['save_interval'] > 0 and (itr % v['save_interval'] == 0 or itr == v['n_itrs']-1):
            torch.save(policy.actor.state_dict(), os.path.join(logger.get_dir(), f"model/policy_{itr}.pkl"))

        logger.dump_tabular()

if __name__ == "__main__":
    print('Entered code')
    main()
