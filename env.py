import numpy as np
import gym
import sys
import random
import time
import inspect
import os
import copy
import itertools
import string
import imageio
from collections import deque
from IPython import embed
from io import StringIO
from gym import Env, spaces
from gym.utils import seeding
if __package__ == '':
    from utils import four_directions, discount_cumsum, Render, chunk, extract
else:
    from .utils import four_directions, discount_cumsum, Render, chunk, extract
import matplotlib.pyplot as plt
from IPython import embed

B = 10000000
SPAN = 1
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
GET = 4
ACTION_NAMES= ['UP', 'DOWN', 'LEFT', 'RIGHT', 'GET']

def color_interpolate(x, start_color, end_color):
    assert ( x <= 1 ) and ( x >= 0 )
    if not isinstance(start_color, np.ndarray):
        start_color = np.asarray(start_color[:3])
    if not isinstance(end_color, np.ndarray):
        end_color = np.asarray(end_color[:3])
    return np.round( (x * end_color + (1 - x) * start_color) * 255.0 ).astype(np.uint8)

CUR_DIR = os.path.dirname(__file__)
ACT_DICT = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
render_map_funcs = {
              '%': lambda x: color_interpolate(x, np.array([0.3, 0.3, 0.3]), np.array([.5, .5, .5])),
              ' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.2)),
              '#': lambda x: color_interpolate(x, np.array([73, 49, 28]) / 255.0, np.array([219, 147, 86]) / 255.0),
              #'%': lambda x: color_interpolate(x, np.array([0.3, 0.3, 0.3]), np.array([.3, .3, .3])),
              #' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.02)),
              #'#': lambda x: color_interpolate(x, np.array([219, 147, 86]) / 255.0, np.array([219, 147, 86]) / 255.0),
              'A': lambda x: (np.asarray(plt.cm.Reds(0.8)[:3]) * 255 ).astype(np.uint8),
              'B': lambda x: (np.asarray(plt.cm.Blues(0.8)[:3]) * 255 ).astype(np.uint8),
              'C': lambda x: (np.asarray(plt.cm.Greens(0.8)[:3]) * 255 ).astype(np.uint8),
              'D': lambda x: (np.asarray(plt.cm.Wistia(0.8)[:3]) * 255 ).astype(np.uint8),
              'E': lambda x: (np.asarray(plt.cm.Purples(0.8)[:3]) * 255 ).astype(np.uint8)
             }# hard-coded

render_map = { k: render_map_funcs[k](1) for k in render_map_funcs }

def construct_render_map(vc):
    np_random = np.random.RandomState(9487)
    pertbs = dict()
    for i in range(20):
        pertb = dict()
        for c in render_map:
            if vc:
                pertb[c] = render_map_funcs[c](np_random.uniform(0, 1))
            else:
                pertb[c] = render_map_funcs[c](0)
        pertbs[i] = pertb
    return pertbs

# TODO: need_get, hindsight

def read_map(filename):
    m = []
    with open(filename) as f:
        for row in f:
            m.append(list(row.rstrip()))
    return m

def dis(a, b, p=2):
    res = 0
    for i, j in zip(a, b):
        res += np.power(np.abs(i-j), p)
    return np.power(res, 1.0/p)

def not_corner(m, i, j):
    if i == 0 or i == len(m)-1 or j == 0 or j == len(m[0])-1:
        return False
    if m[i-1][j] == '#' or m[i+1][j] == '#' or m[i][j-1] == '#' or m[i][j+1] == '#':
        return False
    return True

def build_gaussian_grid(grid, mean, std_coeff):
    row, col = grid.shape
    x, y = np.meshgrid(np.arange(row), np.arange(col))
    d = np.sqrt(x*x + y*y)
    return np.exp(-((x-mean[0]) ** 2 + (y-mean[1]) ** 2)/(2.0 * (std_coeff * min(row, col)) ** 2))

# roll list when task length is 2
def roll_list(l, n):
    res = []
    l = list(chunk(l, n-1))
    for i in range(n-1):
        for j in range(n):
            res.append(l[j][(i+j)%(n-1)])
    return res

class GridWorld(Env):
    metadata = {'render.modes': ['human', 'ansi']}
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
            need_get=True,
            seed=0):
        self.seed(seed)
        self.map_names = map_names
        self.maps = [read_map(os.path.join(CUR_DIR, 'maps', '{}.txt'.format(m))) for m in map_names]
        self.num_obj_types = num_obj_types
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
        self.train_combos = train_combos
        self.test_combos = test_combos
        self.n_train_combos = len(train_combos)
        self.n_test_combos = len(test_combos)
        self.img_stack = deque(maxlen=window)
        self.window = window
        self.gaussian_img = gaussian_img
        self.distance = dict()
        if reward_config is None:
            reward_config = {'wall_penalty': -0.01, 'time_penalty': -0.01, 'complete_sub_task': 1, 'complete_all': 10, 'fail': -10}
        self.reward_config = reward_config
        self.need_get = need_get
        self.observation_space = spaces.Box(low=-B, high=B, shape=(6+2*self.num_obj_types,))
        self.action_space = spaces.Discrete(5) if need_get else spaces.Discrete(4)
        # scene, task
        self.row, self.col = len(self.maps[0]), len(self.maps[0][0]) # make sure all the maps you load are of the same size
        self.m = None
        self.task = None
        self.map_id = None
        self.task_id = None
        self.last_action = None
        self.last_reward = 0.0
        self._render = None

    def seed(self, seed=None):
        self.random, seed = seeding.np_random(seed)
        return seed

    def sample(self, train=True):
        if train:
            index = self.train_combos[self.random.randint(self.n_train_combos)]
        else:
            index = self.test_combos[self.random.randint(self.n_test_combos)]
        self.set_index(index)

    def set_index(self, index):
        self.map_id, self.task_id = index
        self.m = copy.deepcopy(self.maps[self.map_id])
        self.task = np.asarray(copy.deepcopy(self.tasks[self.task_id]))
        self.build_graph() # for optimal planner

    def _set_up_map(self, sample_pos):
        self.mask = np.ones(self.num_obj_types, dtype=np.uint8)
        self.wall = np.zeros((self.row, self.col))
        self.pos_candidates = [] # for object and task
        for i in range(len(self.m)):
            for j in range(len(self.m[i])):
                if self.m[i][j] == '@':
                    self.x = i
                    self.y = j
                    self.m[i][j] = ' '
                elif self.m[i][j] == '#':
                    self.wall[i][j] = 1
                if self.m[i][j] == ' ': #and not_corner(self.m, i, j):
                    self.pos_candidates.append((i, j))
        if sample_pos:
            self.x, self.y = self.pos_candidates[self.random.randint(len(self.pos_candidates))]
        self.pos = [self.pos_candidates[i] for i in self.random.choice(len(self.pos_candidates), self.num_obj_types, replace=False)]
        for i, p in enumerate(self.pos):
            self.m[p[0]][p[1]] = chr(i + ord('A'))
        self.up = []
        self.down = []
        self.left = []
        self.right = []
        for s in self.m: # distance
            self.up.append(B * np.ones(len(s)))
            self.down.append(B * np.ones(len(s)))
            self.left.append(B * np.ones(len(s)))
            self.right.append(B * np.ones(len(s)))
        for i in range(len(self.m)):
            for j in range(len(self.m[i])):
                if self.m[i][j] == '#':
                    self.up[i][j] = 0
                    self.left[i][j] = 0
                else:
                    if i > 0:
                        self.up[i][j] = self.up[i-1][j] + 1
                    if j > 0:
                        self.left[i][j] = self.left[i][j-1] + 1
        for i in reversed(range(len(self.m))):
            for j in reversed(range(len(self.m[i]))):
                if self.m[i][j] == '#':
                    self.down[i][j] = 0
                    self.right[i][j] = 0
                else:
                    if i < len(self.m) - 1:
                        self.down[i][j] = self.down[i+1][j] + 1
                    if j < len(self.m[i]) - 1:
                        self.right[i][j] = self.right[i][j+1] + 1

    def build_graph(self):
        if self.map_id in self.distance:
            return
        self.points = set()
        distance = dict() # (ux, uy, vx, vy): distance
        self.act = dict()
        for i in range(len(self.m)):
            for j in range(len(self.m[i])):
                if self.m[i][j] != '#':
                    self.points.add((i, j))
        for pos in self.points:
            q = deque()
            q.append(pos)
            distance[pos+pos] = 0
            vis = {pos}
            while q:
                u = q.popleft()
                for v in four_directions(u):
                    if v in self.points and v not in vis:
                        distance[pos+v] = distance[pos+u] + 1
                        if distance[pos+u] == 0:
                            self.act[pos+v] = ACT_DICT[(v[0]-u[0], v[1]-u[1])]
                        else:
                            self.act[pos+v] = self.act[pos+u]
                        q.append(v)
                        vis.add(v)
        self.distance[self.map_id] = distance

    def reset(self, index=None, sample_pos=False, train=True):
        if index is None:
            self.sample(train=train)
        else:
            self.set_index(index)
        self._set_up_map(sample_pos)
        for _ in range(self.img_stack.maxlen):
            self.img_stack.append(np.zeros((2+self.num_obj_types, self.row, self.col)))
        self.img_stack.append(self.get_img())
        return self.get_obs()

    # calculte observation
    def get_obs(self):
        out = np.zeros(self.observation_space.shape[0])
        out[0] = self.x
        out[1] = self.y
        out[2] = self.up[self.x][self.y]
        out[3] = self.down[self.x][self.y]
        out[4] = self.left[self.x][self.y]
        out[5] = self.right[self.x][self.y]
        for i, p in enumerate(self.pos):
            if self.mask[i]:
                d = 1 + dis(p, (self.x, self.y))
            else:
                d = B
            out[6+i] = 1.0 / d
        out[6+self.num_obj_types:] = self.mask
        return out

    def get_img(self, gaussian_factor=0.2):
        img = np.zeros((self.row, self.col, 2+self.num_obj_types))
        if self.gaussian_img:
            img[:,:,0] = build_gaussian_grid(img[:,:,0], np.array([self.x, self.y]), gaussian_factor)
        else:
            img[self.x][self.y][0] = 1
        img[:,:,1] = self.wall
        for i in range(self.num_obj_types):
            if self.mask[i]:
                if self.gaussian_img:
                    img[:,:,2+i] = build_gaussian_grid(img[:,:,2+i], np.array([self.pos[i][0], self.pos[i][1]]), gaussian_factor)
                else:
                    img[self.pos[i][0]][self.pos[i][1]][2+i] = 1
        return img.transpose(2, 0, 1)

    def get_imgs(self):
        return np.concatenate(self.img_stack)

    @property
    def task_idx(self):
        return self.num_obj_types - self.mask.sum(dtype=np.uint8)

    def valid_task(self, task):
        task = list(task)
        return self.task_idx < self.task_length and np.all(self.mask[task[:self.task_idx]] == 0)

    # return reward and done or not
    def process_get(self, task):
        r = self.reward_config['time_penalty']
        done = False
        c = ord(self.m[self.x][self.y]) - ord('A')
        idx = self.task_idx
        if c == task[idx]: # correct order
            r += self.reward_config['complete_sub_task']
            if idx + 1 == self.task_length:
                r += self.reward_config['complete_all']
                done = True
        else:
            r += self.reward_config['fail']
            done = True
        return c, r, done

    def teleport(self, x, y):
        self.x = x
        self.y = y
        self.img_stack.append(self.get_img())
        return (self.get_obs(), 0.0, False, {})

    def step(self, action):
        r = self.reward_config['time_penalty']
        done = False
        self.last_action = ACTION_NAMES[action]
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
            c, r, done = self.process_get(self.task) # not adding to previous r
            self.m[self.x][self.y] = ' '
            self.mask[c] = 0
        self.last_reward = r # debug
        self.img_stack.append(self.get_img())
        return (self.get_obs(), r, done, {})

    def render(self, mode='human', close=False, verbose=True):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        if verbose:
            out = 'scene: {}, task: {}, index: {}\n'.format(self.map_id, self.task_desc[self.task_id], self.index())
            out += 'last action: {}\n'.format(self.last_action) if self.last_action is not None else ''
            obs = self.get_obs()
            out += 'pos: ({}, {})\n'.format(obs[0], obs[1])
            out += 'wall distance: {}\n'.format(obs[2:6])
            out += 'distance to all goals: {}\n'.format(obs[6:6+self.num_obj_types])
            out += 'mask: {}\n'.format(obs[6+self.num_obj_types:])
            out += 'last reward: {}\n'.format(self.last_reward)
        else:
            out = ''
        for x in range(len(self.m)):
            for y in range(len(self.m[x])):
                if x == self.x and y == self.y:
                    out += '%'
                else:
                    out += self.m[x][y]
            out += "\n"
        outfile.write(out)
        if mode != 'human':
            return outfile

    def init_render(self):
      if self._render is None:
          self._render = Render()
      return self

    # color table: https://www.rapidtables.com/web/color/RGB_Color.html
    def pretty_render(self, init_render=False, repeat=15):
        out = self.render(verbose=False, mode='ansi')
        world = np.zeros((self.row, self.col, 3))
        i, j = 0, 0
        for c in out.getvalue():
            if c == '\n':
                i += 1
                j = 0
            else:
                world[i, j, :] = render_map[c]
                j += 1
        world = world.repeat(repeat, 0).repeat(repeat, 1)
        if init_render:
            self.init_render()
            self._render.render(world)
        return world

    def render_map(self):
        out = self.render(verbose=False, mode='ansi')
        world = np.zeros((self.row, self.col, 3))
        i, j = 0, 0
        for c in out.getvalue():
            if c == '\n':
                i += 1
                j = 0
            else:
                if c not in ['#', ' ']: c = ' '
                world[i, j, :] = render_map[c]
                j += 1
        world = world.repeat(15, 0).repeat(15, 1)
        return world

    def get_opt_action(self, task=None):
        if task is None:
            task = self.task
        pos = (self.x, self.y)
        dst = self.pos[task[self.task_idx]]
        if pos == dst:
            return 4
        return self.act[pos+dst]

    def get_random_opt_action(self, discount, task=None):
        qs = self.get_qs(discount, task=task)
        best_actions = []
        max_q = np.max(qs)
        for i, q in enumerate(qs):
            if max_q - q < 1e-8:
                best_actions.append(i)
        a = np.random.choice(best_actions)
        return a

    # can specify position, using current task, it is actually value function
    def get_q(self, discount, pos=None, task=None):
        if task is None:
            task = self.task
        if not self.valid_task(task):
            return 0
        if pos is None:
            pos = (self.x, self.y)
        q = self.reward_config['complete_all']
        d = -1
        time = 0
        poses = [pos] + [self.pos[t] for t in task[self.task_idx:]]
        for i in range(len(poses)-1, 0, -1):
            q *= discount ** (d + 1)
            q += self.reward_config['complete_sub_task']
            d = self.distance[self.map_id][poses[i-1]+poses[i]]
            time += 1 + d
        q *= discount ** d # the first step
        q += self.reward_config['time_penalty'] * (1 - discount ** time) / (1 - discount)
        return q

    def get_qs(self, discount, task=None):
        if task is None:
            task = self.task
        qs = []
        for x, y in four_directions((self.x, self.y)):
            if x < 0 or x >= self.row or y < 0 or y >= self.col or self.m[x][y] == '#':
                qs.append(self.reward_config['time_penalty'] + self.reward_config['wall_penalty'] + discount * self.get_q(discount, (self.x, self.y), task=task))
            else:
                qs.append(self.reward_config['time_penalty'] + discount * self.get_q(discount, (x, y), task=task))
        if self.m[self.x][self.y].isalpha():
            if self.m[self.x][self.y] != chr(task[self.task_idx]+ord('A')):
                qs.append(self.reward_config['time_penalty'] + self.reward_config['fail'])
            else:
                r = self.reward_config['time_penalty'] + self.reward_config['complete_sub_task']
                idx = self.task_idx
                if idx + 1 == self.task_length:
                    r += self.reward_config['complete_all']
                else:
                    self.mask[task[idx]] = 0
                    r += discount * self.get_q(discount, task=task)
                    self.mask[task[idx]] = 1
                qs.append(r)
        else:
            qs.append(self.reward_config['time_penalty'] + discount * self.get_q(discount, (self.x, self.y), task=task))
        return qs

    def index(self):
        return self.map_id, self.task_id

class EnvWrapper(gym.Wrapper):
    def pretty_render(self):
        return self.env.unwrapped.pretty_render()

    def get_opt_action(self):
        return self.env.unwrapped.get_opt_action()

    def get_random_opt_action(self, discount):
        return self.env.unwrapped.get_random_opt_action(discount)

    def index(self):
        return self.env.unwrapped.index()

    def get_q(self, *args, **kwargs):
        return self.env.unwrapped.get_q(*args, **kwargs)

    def get_qs(self, *args, **kwargs):
        return self.env.unwrapped.get_qs(*args, **kwargs)

    def get_img(self, gaussian_factor=0.2):
        return self.env.get_img(guassian_factor)

    def get_imgs(self):
        return self.env.get_imgs()

    @property
    def map_names(self):
        return self.env.unwrapped.map_names

    @property
    def tasks(self):
        return self.env.unwrapped.tasks

    @property
    def task_desc(self):
        return self.env.unwrapped.task_desc

    @property
    def window(self):
        return self.env.unwrapped.window

class ComboEnv(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        row, col = self.env.unwrapped.row, self.env.unwrapped.col
        num_obj_types = self.env.unwrapped.num_obj_types
        self.observation_space = spaces.Tuple((spaces.Box(-B, B, (env.window*(2+num_obj_types), row, col)), spaces.Box(-B, B, (6+2*num_obj_types,))))
        self.action_space = spaces.Discrete(5)

    def reset(self, *args, **kwargs):
        o = self.env.reset(*args, **kwargs)
        return (self.env.get_imgs(), o)

    def step(self, action):
        next_o, r, done, info = self.env.step(action)
        return (self.env.get_imgs(), next_o), r, done, info

# on top of ComboEnv
class PORGBEnv(EnvWrapper):
    def __init__(self, env, l=1, vc=False, record=False):
        super().__init__(env)
        self.row, self.col = self.env.unwrapped.row, self.env.unwrapped.col
        window = self.env.unwrapped.window
        num_obj_types = self.env.unwrapped.num_obj_types
        self.img_stack = deque(maxlen=window)
        self.observation_space = spaces.Tuple((spaces.Box(-B, B, (3*window, self.row, self.col)), spaces.Box(-B, B, (6+2*num_obj_types,))))
        self.action_space = spaces.Discrete(5)
        self.l = l
        self.pertbs = construct_render_map(vc)
        self.imgs = [] if record else None

    def _get_obs(self, o):
        if self.imgs is not None:
            self.imgs.append(self.pretty_render())
        return (self._get_imgs(), o[1])

    def _get_imgs(self):
        return np.concatenate(self.img_stack)

    def reset(self, *args, **kwargs):
        self.imgs = []
        o = self.env.reset(*args, **kwargs)
        for i in range(self.img_stack.maxlen):
            self.img_stack.append(np.zeros((3, self.row, self.col)))
        self.img_stack.append(self.generate_img())
        return self._get_obs(o)

    def teleport(self, x, y):
        next_o, r, done, info = self.env.teleport(x, y)
        self.img_stack.append(self.generate_img())
        return self._get_obs(next_o), r, done, info

    def step(self, action):
        next_o, r, done, info = self.env.step(action)
        self.img_stack.append(self.generate_img())
        return self._get_obs(next_o), r, done, info

    def generate_img(self):
        img = np.zeros((3, self.row, self.col))
        pertb = self.pertbs[self.env.unwrapped.map_id]
        m = self.env.unwrapped.m

        # Obtain object center and sight
        ax, ay = self.env.unwrapped.x, self.env.unwrapped.y
        x0 = max(0, ax - self.l)
        x1 = min(self.row - 1, ax + self.l)
        y0 = max(0, ay - self.l)
        y1 = min(self.col - 1, ay + self.l)
        for x in range(len(self.env.unwrapped.m)):
            for y in range(len(self.env.unwrapped.m[x])):
                if x == ax and y == ay: # agent position
                    img[:, x, y] = pertb['%']
                elif m[x][y] != ' ' and m[x][y] != '#': # anything other than 'road' and 'wall'
                    img[:, x, y] = pertb[m[x][y]]
                elif (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
                    img[:, x, y] = pertb[m[x][y]]

        return img

    def render(self):
        self.env.unwrapped.init_render()
        self.env.unwrapped._render.render(self.img_stack[-1].transpose(1,2,0).repeat(16, 0).repeat(16, 1))

    def pretty_render(self, repeat=16):
        return self.img_stack[-1].transpose(1,2,0).repeat(repeat, 0).repeat(repeat, 1)

    def save_record(self, path):
        assert self.imgs is not None, 'does not support record'
        imageio.mimsave(path, self.imgs, 'GIF', duration=0.2)

    # input: random seed before reset, and the action sequence
    # output: a gif
    def dump_traj(self, rng, actions, filepath):
        self.unwrapped.random = copy.deepcopy(rng)
        self.reset(sample_pos=True)
        for a in actions:
            self.step(a)
        self.save_record(filepath)
