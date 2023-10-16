import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces


class ReacherGCEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "reacher.xml", 2)
        obs = self.reset()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            ))

    def reset(self, new_goal=True):
        self.sim.reset()
        obs = self.reset_model(new_goal)
        return obs

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        # reward_dist = -np.linalg.norm(vec)
        # reward_ctrl = -np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        reward = 0.0
        sucess = False
        if np.linalg.norm(vec) < 0.02:
            sucess = True
            reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {'is_success':sucess}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self, new_goal=True):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        if new_goal:
            while True:
                self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
                if np.linalg.norm(self.goal) < 0.2:
                    break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        observation = np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
        ])

        return {
            "observation": observation.copy(),
            "achieved_goal": self.get_body_com("fingertip").copy()[:-1],
            "desired_goal": self.get_body_com("target")[:-1],
        }
        # return np.concatenate(
        #     [
        #         np.cos(theta),
        #         np.sin(theta),
        #         self.sim.data.qpos.flat[2:],
        #         self.sim.data.qvel.flat[:2],
        #         self.get_body_com("fingertip"),
        #         self.get_body_com("target"),
        #     ]
        
