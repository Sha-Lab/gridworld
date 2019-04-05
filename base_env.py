import numpy as np
import gym
import os
import string
import copy
import matplotlib.pyplot as plt
from collections import deque
from gym import Env, spaces
from gym.utils import seeding

if __package__ == '':
    from utils import four_directions, Render, np_sample
else:
    from .utils import four_directions, Render, np_sample

B = 10000000
CUR_DIR = os.path.dirname(__file__)
ACT_DICT = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

### color setup for gridworld ###
def color_interpolate(x, start_color, end_color):
    assert ( x <= 1 ) and ( x >= 0 )
    if not isinstance(start_color, np.ndarray):
        start_color = np.asarray(start_color[:3])
    if not isinstance(end_color, np.ndarray):
        end_color = np.asarray(end_color[:3])
    return np.round( (x * end_color + (1 - x) * start_color) * 255.0 ).astype(np.uint8)

render_map_funcs = {
              '%': lambda x: color_interpolate(x, np.array([0.3, 0.3, 0.3]), np.array([.5, .5, .5])),
              ' ': lambda x: color_interpolate(x, plt.cm.Greys(0.02), plt.cm.Greys(0.2)),
              '#': lambda x: color_interpolate(x, np.array([73, 49, 28]) / 255.0, np.array([219, 147, 86]) / 255.0),
             }

render_map_funcs.update({
    'A': lambda x: (255 * np.array((0.7364705882352941, 0.08, 0.10117647058823528))).astype(np.uint8), 
    'B': lambda x: (255 * np.array((0.09019607843137256, 0.39294117647058824, 0.6705882352941177))).astype(np.uint8), 
    'C': lambda x: (255 * np.array((0.0823529411764706, 0.49803921568627446, 0.23137254901960783))).astype(np.uint8), 
    'D': lambda x: (255 * np.array((0.9976470588235294, 0.6015686274509804, 0.0))).astype(np.uint8), 
    'E': lambda x: (255 * np.array((0.3811764705882353, 0.25176470588235295, 0.6078431372549019))).astype(np.uint8), 
    'F': lambda x: (255 * np.array((0.7587183008012618, 0.7922069335474338, 0.9543861221913403))).astype(np.uint8), 
    'G': lambda x: (255 * np.array((1.0, 1.0, 0.6000000000000001))).astype(np.uint8), 
    'H': lambda x: (255 * np.array((0.19999999999999996, 0.19999999999999996, 0.19999999999999996))).astype(np.uint8), 
    'I': lambda x: (255 * np.array((1.0, 0.4, 0.4))).astype(np.uint8), 
    'J': lambda x: (255 * np.array((0.4, 0.6, 0.0))).astype(np.uint8), 
    'K': lambda x: (255 * np.array((0.9, 0.81, 0.26))).astype(np.uint8), 
    'L': lambda x: (255 * np.array((0.8, 0.19999999999999996, 1.0))).astype(np.uint8), 
    'M': lambda x: (255 * np.array((0.0, 0.0, 0.6173258487801682))).astype(np.uint8), 
    'N': lambda x: (255 * np.array((0.40000000000000036, 0.7000000000000002, 0.8))).astype(np.uint8), 
    'O': lambda x: (255 * np.array((0.7405812349888883, 1.0, 0.0))).astype(np.uint8), 
    'P': lambda x: (255 * np.array((0.6, 0.488, 0.46399999999999997))).astype(np.uint8), 
    'Q': lambda x: (255 * np.array((0.0, 0.8, 0.6))).astype(np.uint8), 
    'R': lambda x: (255 * np.array((0.207843137254902, 0.592156862745098, 0.5607843137254902))).astype(np.uint8), 
    'S': lambda x: (255 * np.array((0.4980392156862745, 0.7372549019607844, 0.2549019607843137))).astype(np.uint8), 
    'T': lambda x: (255 * np.array((0.6007843137254902, 0.00392156862745098, 0.4831372549019608))).astype(np.uint8), 
    'U': lambda x: (255 * np.array((0.16304347826086973, 0.0, 1.0))).astype(np.uint8), 
    'V': lambda x: (255 * np.array((0.3999999999999999, 0.0, 0.0))).astype(np.uint8), 
    'W': lambda x: (255 * np.array((1.0, 0.19999999999999996, 0.0))).astype(np.uint8), 
    'X': lambda x: (255 * np.array((0.4, 0.4, 1.0))).astype(np.uint8), 
    'Y': lambda x: (255 * np.array((0.19999999999999996, 0.8, 1.0))).astype(np.uint8), 
    'Z': lambda x: (255 * np.array((0.0, 0.30000000000000004, 1.0))).astype(np.uint8),
})

def construct_render_map(vc):
    np_random = np.random.RandomState(9487)
    pertbs = dict()
    for i in range(20):
        pertb = dict()
        for c in render_map_funcs:
            if vc:
                pertb[c] = render_map_funcs[c](np_random.uniform(0, 1))
            else:
                pertb[c] = render_map_funcs[c](0)
        pertbs[i] = pertb
    return pertbs

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

# input: a map
# output: the distance dictionary and act dictionary
def build_graph(m):
    points = set()
    distance = dict() # (ux, uy, vx, vy): distance
    act = dict()
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] != '#':
                points.add((i, j))
    for pos in points:
        q = deque()
        q.append(pos)
        distance[pos+pos] = 0
        vis = {pos}
        while q:
            u = q.popleft()
            for v in four_directions(u):
                if v in points and v not in vis:
                    distance[pos+v] = distance[pos+u] + 1
                    if distance[pos+u] == 0:
                        act[pos+v] = ACT_DICT[(v[0]-u[0], v[1]-u[1])]
                    else:
                        act[pos+v] = act[pos+u]
                    q.append(v)
                    vis.add(v)
    return distance, act

# Base Abstract Class
# Based on the observation space, it only contains, agent, wall and objects
class GridWorld(Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(
            self,
            map_names,
            num_obj_types=5,
            train_combos=None,
            test_combos=None,
            window=1,
            gaussian_img=True,
            seed=0):
        self.seed(seed)
        self.map_names = map_names
        self.maps = [read_map(os.path.join(CUR_DIR, 'maps', '{}.txt'.format(m))) for m in map_names]
        self.num_obj_types = num_obj_types
        self.img_stack = deque(maxlen=window)
        self.train_combos = train_combos
        self.test_combos = test_combos
        self.n_train_combos = len(train_combos)
        self.n_test_combos = len(test_combos)
        self.window = window
        self.gaussian_img = gaussian_img
        self.distance = dict()
        self.act = dict()
        self.row, self.col = len(self.maps[0]), len(self.maps[0][0]) # make sure all the maps you load are of the same size
        self.observation_space = spaces.Box(low=-B, high=B, shape=(2 + self.num_obj_types, self.row, self.col)) # agent, wall, objects
        # action space not defined yet
        self.m = None
        self.map_id = None
        self._render = None

    def get_map(self, idx_or_name):
        if isinstance(idx_or_name, int):
            return copy.deepcopy(self.maps[idx_or_name])
        elif isinstance(idx_or_name, str):
            return copy.deepcopy(self.maps[self.map_names.index(idx_or_name)])
        else:
            raise Exception('input type should be int or str')

    def sample(self, train=True):
        if train:
            index = self.train_combos[self.random.randint(self.n_train_combos)]
        else:
            index = self.test_combos[self.random.randint(self.n_test_combos)]
        self.set_index(index)

    def set_index(self, index):
        raise NotImplementedError

    # setup mask, wall, initial agent position and pos_candidates
    def _set_up_map(self):
        self.x, self.y = None, None
        # mask needed to be updated in actual step
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
        assert self.x is not None, 'initial position @ not specified in map'

    def seed(self, seed=None):
        self.random, seed = seeding.np_random(seed)
        return seed

    @property
    def agent_pos(self):
        return (self.x, self.y)

    def build_graph(self):
        if self.map_id in self.distance:
            return
        self.distance[self.map_id], self.act[self.map_id] = build_graph(self.m)

    # used to return observation
    def get_obs(self):
        return np.concatenate(self.img_stack)

    def _get_obs(self, gaussian_factor=0.2):
        img = np.zeros((self.row, self.col, 2+self.num_obj_types))
        if self.gaussian_img:
            img[:,:,0] = build_gaussian_grid(img[:,:,0], np.array([self.x, self.y]), gaussian_factor)
        else:
            img[self.x][self.y][0] = 1
        img[:,:,1] = self.wall
        for i in range(self.num_obj_types):
            if self.mask[i]:
                if self.gaussian_img:
                    img[:,:,2+i] = build_gaussian_grid(img[:,:,2+i], np.array([self.object_pos[i][0], self.object_pos[i][1]]), gaussian_factor)
                else:
                    img[self.object_pos[i][0]][self.object_pos[i][1]][2+i] = 1
        return img.transpose(2, 0, 1)

    def reset(self, index=None, sample_pos=True, train=True):
        if index is None:
            self.sample(train=train)
        else:
            self.set_index(index)
        self._set_up_map(sample_pos=sample_pos) # how to setup map is env dependent
        for _ in range(self.img_stack.maxlen):
            #self.img_stack.append(np.zeros((2+self.num_obj_types, self.row, self.col)))
            self.img_stack.append(np.zeros(self.observation_space.shape))
        self.img_stack.append(self._get_obs())
        return self.get_obs()

    def step(self, action):
        raise NotImplementedError

    def get_info(self):
        return dict(pos=(self.x, self.y))

    def teleport(self, x, y):
        self.x = x
        self.y = y
        self.img_stack.append(self._get_obs())
        return self.get_obs(), 0.0, False, self.get_info()

    # color table: https://www.rapidtables.com/web/color/RGB_Color.html
    def render(self, mode='human', close=False):
        raise NotImplementedError

    def init_render(self):
      if self._render is None:
          self._render = Render(size=(self.col * 16, self.row * 16))
      return self

    def get_opt_action(self):
        raise NotImplementedError

    def get_random_opt_action(self, discount):
        raise NotImplementedError

    # can specify position
    def get_v(self, discount):
        raise NotImplementedError

    def get_q(self, discount):
        raise NotImplementedError

class EnvWrapper(gym.Wrapper):
    def pretty_render(self):
        return self.env.unwrapped.pretty_render()

    def get_opt_action(self):
        return self.env.unwrapped.get_opt_action()

    def get_random_opt_action(self, discount):
        return self.env.unwrapped.get_random_opt_action(discount)

    def index(self):
        return self.env.unwrapped.index()

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        return self.env.get_obs()

    def get_v(self, *args, **kwargs):
        return self.env.unwrapped.get_v(*args, **kwargs)

    def get_q(self, *args, **kwargs):
        return self.env.unwrapped.get_q(*args, **kwargs)

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

    @property
    def success_reward(self):
        return self.env.unwrapped.success_reward

    # warning: this will change the state of the environment
    # iterate over all pos_cancidates (teleportable states) and retrieve the state and info
    def get_teleportable_states(self, discount, index=None):
        self.reset(index=index) # this will change the state! (randomly sample from train)
        states = []
        infos = []
        for x, y in self.env.unwrapped.pos_candidates:
            o, _, _, info = self.teleport(x, y)
            info['opt_a'] = self.get_opt_action()
            info['v'] = self.get_v(discount=discount)
            info['q'] = self.get_q(discount=discount)
            states.append(o)
            infos.append(info)
        return states, infos

# set a large window size can become fully observable
class PORGBEnv(EnvWrapper):
    def __init__(self, env, l=3, vc=False, record=False):
        super().__init__(env)
        self.row, self.col = self.env.unwrapped.row, self.env.unwrapped.col
        window = self.env.unwrapped.window
        num_obj_types = self.env.unwrapped.num_obj_types
        self.img_stack = deque(maxlen=window)
        self.observation_space = spaces.Box(-B, B, (3*window, self.row, self.col))
        self.action_space = self.env.unwrapped.action_space
        self.l = l
        self.pertbs = construct_render_map(vc)
        self.imgs = [] if record else None

    def update_obs(self):
        self.img_stack.append(self._get_obs())
        if self.imgs is not None: # for recording
            self.imgs.append(self.pretty_render())

    def get_obs(self):
        return np.concatenate(self.img_stack)

    def reset(self, *args, **kwargs):
        self.imgs = []
        o = self.env.reset(*args, **kwargs)
        for i in range(self.img_stack.maxlen):
            self.img_stack.append(np.zeros((3, self.row, self.col)))
        self.update_obs()
        return self.get_obs()

    def teleport(self, x, y):
        next_o, r, done, info = self.env.teleport(x, y)
        self.update_obs()
        return self.get_obs(), r, done, info

    def step(self, action):
        next_o, r, done, info = self.env.step(action)
        self.update_obs()
        return self.get_obs(), r, done, info

    def _get_obs(self):
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
        #imageio.mimsave(path, self.imgs, 'GIF', duration=0.2)
        save_gif(path, [np.rint(img).astype(np.uint8) for img in self.imgs])

    # input: random seed before reset, and the action sequence
    # output: a gif
    # if actions is None, return optimal
    def dump_traj(self, combo, rng, actions, filepath=None, discount=0.99):
        self.unwrapped.random = copy.deepcopy(rng)
        self.reset(combo, sample_pos=True)
        if actions is None:
            done = False
            while not done:
                _, _, done, _ = self.step(np.argmax(self.get_q(discount)))
        else:
            done = False
            for i, a in enumerate(actions):
                _, _, done, _ = self.step(a)
                assert done or i < len(actions) - 1 or i == 299, 'not done in the correct place!' # 300 is the max traj length
        if filepath is not None:
            self.save_record(filepath)
        return self.imgs

class GoalManager:
    def __init__(self, map_name, seed=0):
        self.map = read_map(os.path.join(CUR_DIR, 'maps', '{}.txt'.format(map_name)))
        self.random = np.random.RandomState(seed)
        self.distance, _ = build_graph(self.map)
        self.pos_candidates = [(i, j) for i in range(len(self.map)) for j in range(len(self.map[0])) if self.map[i][j] != '#']

    def gen_goals(self, n_goal, min_dis=1):
        while True:
            pos_candidates = self.pos_candidates
            goals = []
            while True:
                if len(pos_candidates) == 0: break
                pos = np_sample(pos_candidates, random=self.random)
                ok = True
                for g in goals:
                    if self.distance[pos + g] < min_dis:
                        ok = False
                        break
                if ok: goals.append(pos)
                else: pos_candidates.remove(pos)
                if len(goals) == n_goal: return goals

class PosAbstractEnv(EnvWrapper):
    def __init__(self, env, abstractor):
        super().__init__(env)
        self.n_abs = abstractor.n_abs
        self.abs = abstractor

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        info = self.env.get_info()
        return self.abs(info['pos'])

    def step(self, action):
        next_o, r, done, info = self.env.step(action)
        next_o = self.abs(info['pos'])
        return next_o, r, done, info

class Abstractor:
    def __init__(self, n_abs, abs_f):
        self.n_abs = n_abs
        self.abs_f = abs_f

    def __call__(self, s):
        return self.abs_f(s)
