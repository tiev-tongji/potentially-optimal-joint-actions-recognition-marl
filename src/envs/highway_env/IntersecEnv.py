from os import terminal_size
from symbol import term

from matplotlib.dates import num2date
from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
import gym
from utils.dict2namedtuple import convert
import pprint
from gym import spaces
from .envs.intersection_env import IntersectionEnv
import time
import ray


DEFAULT_CONFIG = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "KinematicsWithRoute",
            "vehicles_count": 4,
            "features": ["x", "y", "vx", "vy", "heading"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-200, 200],
                "vy": [-200, 200]
            },
            "absolute": False, # relative to observer_vehicle or not
            "order": "sorted", # sorted by lane distance 
            "normalize": False,
            "clip": False
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            'type': 'DiscreteMetaAction',
            'longitudinal': True,
            'lateral': False,
            "target_speeds": [0, 2, 4, 6, 8, 10] # must have equal gap
        }
    },
    "controlled_vehicles": 4,
    "initial_vehicle_count": 0, # useless
    'spawn_probability': -1., # random vehicles
    "simulation_frequency": 5, # env internal calculation frequency
    "policy_frequency": 5, # env.step() frequency
    "duration": 20, # unit seconds
    'manual_control': False,
    'other_vehicles_type': 'envs.highway_env.vehicle.behavior.IDMVehicle',
    
    'collision_reward': -1,
    'high_speed_reward': 0.2,
    'lane_change_reward': -0.05,
    'right_lane_reward': 0.1,
    'merging_speed_reward': -0.5,
    'ma_arrive_reward': 100,
    'ma_collision_reward': -100,

    'offscreen_rendering': False,
    'real_time_rendering': False, # rendering on simulation_frequency
    'render_agent': True,
    'scaling': 5.5,
    'centering_position': [0.5, 0.5],
    'screen_height': 800,
    'screen_width': 800,
    'show_trajectories': False,

    'seed': 0,
}

class IntersecEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        config = kwargs
        self.env = IntersectionEnv(config)
        self.env.reset()
        
        self.num_agents = self.env.config['controlled_vehicles']
        self.n_actions = 3
        self.single_action_space_size = self.n_actions # depend on action config options lateral and longitudinal
        self.single_action_space = spaces.Discrete(self.single_action_space_size)
        self.single_observation_size = self.env.config['observation']['observation_config']['vehicles_count']*len(self.env.config['observation']['observation_config']['features'])
        if (self.env.config['observation']['observation_config']['type'] == 'KinematicsWithRoute'):
            self.single_observation_size += 10 # route: 5 points (x0, y0, x1, y1 ...)
        self.single_observation_space = spaces.Box(float('-inf'), float('inf'), [self.single_observation_size])
        self.observation_space = {str(i):self.single_observation_space for i in range(self.num_agents)}
        self.action_space = {str(i):self.single_action_space for i in range(self.num_agents)}
        self._agent_ids = set([str(i) for i in range(self.num_agents)])
        self._obs = np.zeros((self.num_agents, self.single_observation_size))
        self.state_shape = self.single_observation_size + (self.num_agents - 1) * 10

        self.episode_limit = self.env.config['duration'] * self.env.config['policy_frequency'] + 2
    
    def reset(self):
        obs, info = self.env.reset()
        self._obs = np.array(obs).reshape(self.num_agents, -1)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        ## self.env.step() params
        # input type: numpy (num_agents)
        # output type: 
        # obs: tuple (num_agents, single_observation_size)
        ## pymarl2 params
        # actions: tensor (num_agents)
        # obs: tensor (num_agents, single_observation_size)
        # avail_actions: tensor (num_agents, single_action_space_size)
        # reward: scalar
        # terminated: scalar
        # info: dict
        obs, reward, terminated, truncated, info = self.env.step(tuple(actions))
        self._obs = np.array(obs).reshape(self.num_agents, -1)
        # when truncated, also terminated
        if truncated:
            info['episode_limit'] = True
        return reward, terminated, info

    def step_to_end(self, mac, t_env, test_mode):
        device = "cpu"
        if device == "cuda":
            mac.cuda(self.args.device)
        env_info = self.get_env_info()
        data = {
            "state": th.zeros((1, self.episode_limit, env_info["state_shape"]), device=device),
            "avail_actions": th.zeros((1, self.episode_limit, env_info["n_agents"], env_info["n_actions"]), device=device),
            "obs": th.zeros((1, self.episode_limit, env_info["n_agents"], env_info["obs_shape"]), device=device),
            "actions": th.zeros((1, self.episode_limit, env_info["n_agents"], 1), device=device),
            "reward": th.zeros((1, self.episode_limit, 1), device=device),
            "terminated": th.zeros((1, self.episode_limit, 1), dtype=th.uint8, device=device),
            "batch_size": 1
        }
        data["state"][0][0] = th.tensor(self.get_state())
        data["obs"][0][0] = th.tensor(self.get_obs())
        data["avail_actions"][0][0] = th.tensor(self.get_avail_actions())
        terminated = False
        self.t = 0
        self.t_env = t_env
        mac.init_hidden(1)
        while not terminated:
            t1 = time.time()
            actions = mac.select_actions(data, t_ep=self.t, t_env=self.t_env, bs=slice(0, 1), test_mode=test_mode)
            t2 = time.time()
            print('select actions time: ', t2 - t1)
            cpu_actions = actions.to("cpu").numpy()
            # cpu_actions = np.zeros((1, self.num_agents)) # for test
            obs, reward, terminated, truncated, info = self.env.step(tuple(cpu_actions[0]))
            self._obs = np.array(obs).reshape(self.num_agents, -1)
            self._state = self.get_state()
            # when truncated, also terminated
            data["state"][0][self.t + 1] = th.tensor(self._state)
            data["avail_actions"][0][self.t + 1] = th.tensor(self.get_avail_actions())
            data["obs"][0][self.t + 1] = th.tensor(obs)
            data["reward"][0][self.t] = reward
            data["terminated"][0][self.t] = terminated
            data["actions"][0][self.t] = th.tensor(cpu_actions.reshape(self.num_agents, 1))
            if truncated:
               data["terminated"][0][-1][0] = False
               break
            if terminated:
                break
            self.t += 1
        # append 1 element to let len(data["obs"][0]) == len(data["termintaed"][0])
        data["reward"][0][self.t] = 0
        data["terminated"][0][self.t] = False
        data["actions"][0][self.t] = th.tensor(np.zeros((self.num_agents, 1)))

        data["state"] = data["state"][:, :(self.t + 1)]
        data["avail_actions"] = data["avail_actions"][:, :(self.t + 1)]
        data["obs"] = data["obs"][:, :(self.t + 1)]
        data["reward"] = data["reward"][:, :(self.t + 1)]
        data["terminated"] = data["terminated"][:, :(self.t + 1)]
        data["actions"] = data["actions"][:, :(self.t + 1)]
        return data

    def get_state(self):
        self.state = np.concatenate((self._obs[:, :len(self.env.config['observation']['observation_config']['features'])], self._obs[:, -10:]), axis=1).flatten()
        return self.state
    
    def get_obs(self):
        # agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return self._obs
    
    def get_avail_actions(self):
        return np.ones((self.num_agents, self.single_action_space_size))

    def get_info(self):
        return self.env._info(np.array([0]), 0)

    def seed(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.state_shape,
                    "obs_shape": self.single_observation_size, # single agent observation (flattened)
                    "n_actions": self.single_action_space.n,
                    "n_agents": self.num_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def close(self):
        return self.env.close()

    def save_replay(self):
        pass

    def render(self):
        self.env.render()

    # for parallel runner
    def get_stats(self):
        pass


@ray.remote
class IntersecEnvRay(IntersecEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)