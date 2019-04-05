import numpy as np
import sys
import random
import time
import inspect
import os
import copy
import itertools
import string
from collections import deque
from io import StringIO
from gym import spaces

if __package__ == '':
    from base_env import GridWorld
    from utils import four_directions, discount_cumsum, Render, chunk, save_gif
else:
    from .base_env import GridWorld
    from .utils import four_directions, discount_cumsum, Render, chunk, save_gif

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
GET = 4
ACTION_NAMES= ['UP', 'DOWN', 'LEFT', 'RIGHT', 'GET']

# roll list when task length is 2
def roll_list(l, n):
    res = []
    l = list(chunk(l, n-1))
    for i in range(n-1):
        for j in range(n):
            res.append(l[j][(i+j)%(n-1)])
    return res

class PickGridWorld(GridWorld):
    def __init__(
            self,
            map_names,
            num_obj_types=5,
            task_length=2,
            train_combos=None,
            test_combos=None,
            window=1,
            gaussian_img=True,
            reward_config=None,
            obj_pos=None, # a list of list of positions same size as map
            min_dis=0,
            seed=0):
        super().__init__(map_names=map_names, num_obj_types=num_obj_types, train_combos=train_combos, test_combos=test_combos, window=window, gaussian_img=gaussian_img, seed=seed)
        self.task_length = task_length
        assert task_length <= num_obj_types, 'task length ({}) should be shorter than number of object types ({})'.format(task_length, num_obj_types)
        self.tasks = list(itertools.permutations(list(range(num_obj_types)), task_length))
        self.task_desc = list(itertools.permutations(list(string.ascii_uppercase[:num_obj_types]), task_length))
        if task_length == 2: # hardcoded preprocess
            self.tasks = roll_list(self.tasks, num_obj_types)
            self.task_desc = roll_list(self.task_desc, num_obj_types)
        print('maps:', list(enumerate(self.map_names)))
        print('tasks:', list(enumerate(self.task_desc)))
        print('train:', train_combos)
        print('test:', test_combos)
        if reward_config is None:
            reward_config = {'wall_penalty': -0.01, 'time_penalty': -0.01, 'complete_sub_task': 1, 'complete_all': 10, 'fail': -10}
        self.reward_config = reward_config
        self.success_reward = reward_config['complete_all'] + reward_config['complete_sub_task'] + reward_config['time_penalty']
        self.action_space = spaces.Discrete(5)
        self.task = None
        self.task_id = None
        self.min_dis = min_dis # minimum trajectory length
        if obj_pos is None:
            obj_pos = [None] * len(map_names)
        self.default_obj_pos = obj_pos
        assert len(self.default_obj_pos) == len(self.map_names)

    def set_index(self, index):
        self.map_id, self.task_id = index
        self.m = copy.deepcopy(self.maps[self.map_id])
        self.task = np.asarray(copy.deepcopy(self.tasks[self.task_id]))
        self.build_graph() # for optimal planner

    def _set_up_map(self, sample_pos=True, sample_obj_pos=True):
        super()._set_up_map()
        if sample_obj_pos:
            self.object_pos = [self.pos_candidates[i] for i in self.random.choice(len(self.pos_candidates), self.num_obj_types, replace=False)]
        else:
            if self.default_obj_pos[self.map_id] is None:
                self.default_obj_pos[self.map_id] = [self.pos_candidates[i] for i in self.random.choice(len(self.pos_candidates), self.num_obj_types, replace=False)]
            self.object_pos = list(self.default_obj_pos[self.map_id])
        for i, p in enumerate(self.object_pos):
            self.m[p[0]][p[1]] = chr(i + ord('A'))
        if sample_pos:
            while True:
                self.x, self.y = self.pos_candidates[self.random.randint(len(self.pos_candidates))] # teleportable states
                dst = self.object_pos[self.task[self.task_idx]]
                if self.distance[self.map_id][(self.x, self.y) + dst] >= self.min_dis: break

    @property
    def task_idx(self):
        return self.num_obj_types - self.mask.sum(dtype=np.uint8)

    def valid_task(self):
        task = list(self.task)
        return self.task_idx < self.task_length and np.all(self.mask[task[:self.task_idx]] == 0)

    # return reward and done or not
    def process_get(self):
        r = self.reward_config['time_penalty']
        done = False
        c = ord(self.m[self.x][self.y]) - ord('A')
        idx = self.task_idx
        if c == self.task[idx]: # correct order
            r += self.reward_config['complete_sub_task']
            if idx + 1 == self.task_length:
                r += self.reward_config['complete_all']
                done = True
        else:
            r += self.reward_config['fail']
            done = True
        return c, r, done

    def reset(self, index=None, sample_pos=True, sample_obj_pos=True, train=True):
        if index is None:
            self.sample(train=train)
        else:
            self.set_index(index)
        self._set_up_map(sample_pos=sample_pos, sample_obj_pos=sample_obj_pos)
        for _ in range(self.img_stack.maxlen):
            self.img_stack.append(np.zeros(self.observation_space.shape))
        self.img_stack.append(self._get_obs())
        return self.get_obs()

    def step(self, action):
        r = self.reward_config['time_penalty']
        done = False
        if action == 0:
            if self.x > 0:
                if self.m[self.x-1][self.y] != '#':
                    self.x -= 1
                else:
                    r += self.reward_config['wall_penalty']
        elif action == 1:
            if self.x < len(self.m)-1:
                if self.m[self.x+1][self.y] != '#':
                    self.x += 1
                else:
                    r += self.reward_config['wall_penalty']
        elif action == 2:
            if self.y > 0:
                if self.m[self.x][self.y-1] != '#':
                    self.y -= 1
                else:
                    r += self.reward_config['wall_penalty']
        elif action == 3:
            if self.y < len(self.m[self.x])-1:
                if self.m[self.x][self.y+1] != '#':
                    self.y += 1
                else:
                    r += self.reward_config['wall_penalty']
        elif self.m[self.x][self.y].isalpha():
            c, r, done = self.process_get() # not adding to previous r
            self.m[self.x][self.y] = ' '
            self.mask[c] = 0
        self.img_stack.append(self._get_obs())
        return self.get_obs(), r, done, self.get_info()

    # pos is the meta state that used to do state abstraction extraction
    def get_info(self):
        return dict(map_id=self.map_id, task_id=self.task_id, pos=(self.x, self.y))

    # color table: https://www.rapidtables.com/web/color/RGB_Color.html
    def render(self, mode='human', close=False):
        raise NotImplementedError

    def get_opt_action(self):
        pos = self.agent_pos
        dst = self.object_pos[self.task[self.task_idx]]
        if pos == dst:
            return 4
        return self.act[self.map_id][pos+dst]

    def get_random_opt_action(self, discount):
        qs = self.get_q(discount)
        best_actions = []
        max_q = np.max(qs)
        for i, q in enumerate(qs):
            if max_q - q < 1e-8:
                best_actions.append(i)
        a = np.random.choice(best_actions)
        return a

    # can specify position, using current task, it is actually value function
    def get_v(self, discount, pos=None):
        if not self.valid_task():
            return 0
        if pos is None:
            pos = self.agent_pos
        q = self.reward_config['complete_all']
        d = -1
        time = 0
        poses = [pos] + [self.object_pos[t] for t in self.task[self.task_idx:]]
        for i in range(len(poses)-1, 0, -1):
            q *= discount ** (d + 1)
            q += self.reward_config['complete_sub_task']
            d = self.distance[self.map_id][poses[i-1]+poses[i]]
            time += 1 + d
        q *= discount ** d # the first step
        q += self.reward_config['time_penalty'] * (1 - discount ** time) / (1 - discount)
        return q

    def get_q(self, discount):
        task = self.task
        qs = []
        for x, y in four_directions(self.agent_pos):
            if x < 0 or x >= self.row or y < 0 or y >= self.col or self.m[x][y] == '#':
                qs.append(self.reward_config['time_penalty'] + self.reward_config['wall_penalty'] + discount * self.get_v(discount, self.agent_pos))
            else:
                qs.append(self.reward_config['time_penalty'] + discount * self.get_v(discount, (x, y)))
        if self.m[self.x][self.y].isalpha():
            if self.m[self.x][self.y] != chr(self.task[self.task_idx]+ord('A')):
                qs.append(self.reward_config['time_penalty'] + self.reward_config['fail'])
            else:
                r = self.reward_config['time_penalty'] + self.reward_config['complete_sub_task']
                idx = self.task_idx
                if idx + 1 == self.task_length:
                    r += self.reward_config['complete_all']
                else:
                    self.mask[self.task[idx]] = 0
                    r += discount * self.get_v(discount)
                    self.mask[self.task[idx]] = 1
                qs.append(r)
        else:
            qs.append(self.reward_config['time_penalty'] + discount * self.get_v(discount, self.agent_pos))
        return qs

    def test_v(self, discount):
        rs = []
        pos = self.agent_pos
        for t in self.task[self.task_idx:]:
            d = self.distance[self.map_id][pos+self.object_pos[t]]
            rs += [self.reward_config['time_penalty']] * (d + 1)
            rs[-1] += self.reward_config['complete_sub_task']
            pos = self.object_pos[t]
        rs[-1] += self.reward_config['complete_all']
        return discount_cumsum(rs, discount)[0]

    def index(self):
        return self.map_id, self.task_id
