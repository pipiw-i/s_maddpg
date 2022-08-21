# -*- coding: utf-8 -*-
# @Time : 2022/5/17 上午9:52
# @Author :  wangshulei
# @FileName: simple_search.py
# @Software: PyCharm
"""
无人机搜索的环境，在一个方形的环境中尽快的搜索环境，找到目标
"""
import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import copy
import math
import cv2 as cv
from collections import deque


class Scenario(BaseScenario):
    def __init__(self,
                 num_agents=3,  # 160 rgb
                 num_landmarks=0,  # < 100 rgb
                 agent_size=0.15):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.agent_size = agent_size
        # 每个机器人的路径轨迹
        self.traj = []
        self.obs_traj_len = 10
        self.obs_traj = [deque(maxlen=self.obs_traj_len) for _ in range(self.num_agents)]  # 每一个智能体所能记录的自己的轨迹

    def make_world(self):
        world = World()
        world.world_length = 200
        # set any world properties first
        world.dim_c = 2  # 二维
        world.num_agents = self.num_agents
        world.num_landmarks = self.num_landmarks  # 3
        world.collaborative = True  # 是否具有体积
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        self.traj = []
        world.assign_agent_colors()
        world.assign_landmark_colors()
        # set random initial states  随机初始状态
        for agent_index, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.now_research_area = 0
            agent.last_research_area = 0
            agent.last_traj = None
            for i in range(self.obs_traj_len):
                self.obs_traj[agent_index].append(copy.deepcopy(agent.state.p_pos))
            agent.last_traj = self.obs_traj[agent_index]
            # 获取智能体每一个点的坐标
            for i in range(30):
                ang = 2 * math.pi * i / 30
                points = np.array([agent.state.p_pos[0] + math.cos(ang) * agent.size,
                                   agent.state.p_pos[1] + math.sin(ang) * agent.size])
                self.traj.append([copy.deepcopy(points)])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_pos = np.array([0.8 * i, 0])
            landmark.state.p_vel = np.zeros(world.dim_p)
        return self.traj

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def calculate_area(self):
        # 计算黑色色块
        img = cv.imread('screenshot.png')
        r = 0
        if img is not None:
            shape = img.shape
            for width in range(0, shape[0], 8):
                for height in range(0, shape[1], 8):
                    if img[width][height][0] == 0 or img[width][height][0] == 160:
                        r += 64
            r = r / (shape[0] * shape[1])
        return r

    def reward(self, agent, world, r):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        agent.now_research_area = r
        rew = (agent.now_research_area - agent.last_research_area) * 200
        print(f"one agent rew is {rew},agent.last_research_area is {agent.last_research_area},"
              f"agent.now_research_area is {agent.now_research_area}")
        agent.last_research_area = agent.now_research_area

        # 不考虑相对于目标点的距离，目前进行探索策略的训练
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
        #              for a in world.agents]
        #     rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        traj = []
        for point in agent.last_traj:
            traj.append(point)
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + traj + comm)
        return obs
