# Wrappers for goal conditioned environments
# Wrapper will 
#   - ensure a maximum number of steps per episode 
#   - change rewards to sparse rewards
#   - convert observations from dictionaries to flat arrays

# Inspired by stable-baselines' HER implementation

import gym
from gym import spaces
import numpy as np
from PIL import Image

KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']

class GCWrapper:
    def __init__(self, env, max_steps=100):
        self.env = env
        self.max_steps = max_steps
        self.current_step = 0
        self.spaces = list(env.observation_space.spaces.values())
        self.action_space = env.action_space

        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)
        self.goal_state_indices = np.arange(self.obs_dim, self.obs_dim + self.goal_dim)

    def reset(self, new_goal=True, init=None, goal=None, get_init=False):
        self.current_step = 0
        # print(self.env)
        obs = self.env.reset(new_goal=new_goal, init=init, goal=goal, get_init=get_init)
        # print("obs: ", obs)
        if type(obs)==tuple:
            return self._get_obs(obs[0]), obs[1]
        return self._get_obs(obs)

    def step(self, action):
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        reward = info['is_success']
        # done = done or info['is_success'] > 0 or self.current_step >= self.max_steps
        done = self.current_step >= self.max_steps
        # done = done or self.current_step >= self.max_steps
        # if done is True:
        #     print(info['is_success'])
        return self._get_obs(obs), float(reward), done, info

    def _get_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        return np.concatenate([obs_dict[key] for key in KEY_ORDER])

    def close(self):
        self.env.close()

    def render(self, save_path):
        rgb_array = self.env.render('rgb_array')
        img = Image.fromarray(rgb_array)
        img.save(save_path)
        