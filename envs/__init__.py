#!/usr/bin/env python

from gym.envs.registration import register
from envs.pointmaze import U_MAZE, MEDIUM_MAZE, LARGE_MAZE, SMALL_MAZE
from envs.pointmaze_tough import U_MAZE_TOUGH, LARGE_MAZE_TOUGH, MEDIUM_MAZE_TOUGH
from envs.locomotion import maze_env
import numpy as np

register(
    id='ReacherDraw-v0',
    entry_point='envs.reacher_trace:ReacherTraceEnv',
)

register(
    id='FetchReachGC-v0',
    entry_point='envs.robotics.fetch.reach:FetchReachEnvGC',
)

register(
    id='ReacherGC-v0',
    entry_point='envs.reacher:ReacherGCEnv',
)

register(
    id='PointMazeU-v0',
    entry_point='envs.pointmaze:MazeEnv',
    kwargs={'maze_spec': U_MAZE}
)

register(
    id='PointMazeMedium-v0',
    entry_point='envs.pointmaze:MazeEnv',
    kwargs={'maze_spec': MEDIUM_MAZE}
)

register(
    id='PointMazeLarge-v0',
    entry_point='envs.pointmaze:MazeEnv',
    kwargs={'maze_spec': LARGE_MAZE}
)

register(
    id='PointMazeSmall-v0',
    entry_point='envs.pointmaze:MazeEnv',
    kwargs={'maze_spec': SMALL_MAZE}
)

register(
    id='PointMazeUTough-v0',
    entry_point='envs.pointmaze_tough:MazeEnvTough',
    kwargs={'maze_spec': U_MAZE_TOUGH}
)

register(
    id='PointMazeMediumTough-v0',
    entry_point='envs.pointmaze_tough:MazeEnvTough',
    kwargs={'maze_spec': MEDIUM_MAZE_TOUGH}
)

register(
    id='PointMazeLargeTough-v0',
    entry_point='envs.pointmaze_tough:MazeEnvTough',
    kwargs={'maze_spec': LARGE_MAZE_TOUGH}
)
