from os import listdir
from itertools import product
import os
import unittest
import dill
import time
import numpy as np
import imageio
import random
from tqdm import tqdm
import readchar
from utils import mkdir, ipdb_on_exception, discount_cumsum, set_seed
from pick_env import PickGridWorld
from base_env import PORGBEnv

HAS_DISPLAY = 'display' in os.environ or 'DISPLAY' in os.environ
CONTROL_MAP = dict(w=0,s=1,a=2,d=3,e=4)

def control(env, save_gif=False):
    assert HAS_DISPLAY
    env.reset(sample_pos=True, sample_obj_pos=False) # debug
    done = False
    discount = 0.99
    imgs = [] # test generate gif
    while not done:
        env.render()
        print(env.unwrapped.get_q(discount))
        assert np.isclose(env.get_v(discount), env.unwrapped.test_v(discount)), 'v and test v not aligned'
        imgs.append(env.pretty_render())
        c = readchar.readchar()
        if c == 'q':
            break
        elif c == 'r':
            env.reset(sample_pos=True, train=True, sample_obj_pos=False) # debug
        elif c == 'i':
            map_id, task_id = input("input scene and task id:").split()
            env.reset((int(map_id), int(task_id)))
        elif c == 'o':
            _, r, done, _ = env.step(env.get_opt_action())
        elif c == 'O':
            _, r, done, _ = env.step(env.get_random_opt_action(discount))
        elif c not in CONTROL_MAP:
            continue
        else:
            qs = env.get_q(discount)
            a = CONTROL_MAP[c]
            o, r, done, info = env.step(a)
            for index, d in info.items():
                print(index, d) # print info
            print(r, done) # debug
            if not done and isinstance(env.unwrapped, PickGridWorld): 
                print('next goal:', env.unwrapped.object_pos[env.unwrapped.task[env.unwrapped.task_idx]]) # for debugging
            diff = abs(r + discount * env.get_v(discount) - qs[a])
            assert diff < 1e-7, 'qs is not consistent! diff: {}'.format(diff)
    if save_gif:
        imageio.mimsave('test.gif', imgs, 'GIF', duration=0.2)

def record_trajs(env, filename, index=None):
    assert HAS_DISPLAY
    from py_tools.common import fsave, Recorder, Storage
    storage = Storage('env')
    env = Recorder(env, call_media=lambda x: storage.push(x.name, dict(input=x.input, output=x.output)), attr_filter=lambda attr: attr in ['step', 'pretty_render', 'get_opt_action', 'get_q'])
    for _ in range(3):
        done = False
        env.reset(index=index, sample_pos=True, train=True)
        env.render()
        # record
        env.pretty_render()
        while not done:
            c = readchar.readchar()
            env.get_opt_action()
            env.get_q(0.99)
            a = CONTROL_MAP[c]
            _, r, done, _ = env.step(a)
            env.render()
            # record
            env.pretty_render() # just for recording
        if r > env.success_reward: print('success!')
        else: print('fail...')
    fsave(storage.get_lists(), filename, ftype='pkl')

def test_trajs(env, filename, index=None):
    from py_tools.common import fload, Storage, Recorder
    storage = Storage('env')
    env = Recorder(env, call_media=lambda x: storage.push(x.name, dict(input=x.input, output=x.output)), attr_filter=lambda attr: attr in ['step', 'pretty_render', 'get_opt_action', 'get_q'])
    env.reset(index=index, sample_pos=True, train=True)
    if HAS_DISPLAY: env.render()
    env.pretty_render()
    origin_records = fload(filename, 'pkl')
    done = False
    for step in origin_records['step']:
        if done:
            env.reset(index=index, sample_pos=True, train=True)
            if HAS_DISPLAY: env.render()
            env.pretty_render()
        env.get_opt_action()
        env.get_q(0.99)
        a = step['input']['action']
        _, _, done, _ = env.step(a)
        if HAS_DISPLAY: env.render()
        env.pretty_render()
    # now compare two records!
    records = storage.get_lists()
    np.testing.assert_equal(origin_records['pretty_render'], records['pretty_render'])
    if isinstance(env.unwrapped, PickGridWorld):
        np.testing.assert_equal(origin_records['get_q'], records['get_q'])
        np.testing.assert_equal([record['output'][0] for record in origin_records['step']], [record['output'][0] for record in records['step']])
    np.testing.assert_equal(origin_records['get_opt_action'], records['get_opt_action'])
    np.testing.assert_equal([record['output'][1] for record in origin_records['step']], [record['output'][1] for record in records['step']])
    np.testing.assert_equal([record['output'][2] for record in origin_records['step']], [record['output'][2] for record in records['step']])

def test_planner(env, discount):
    from tqdm import tqdm
    for i in tqdm(range(200)):
        env.reset(sample_pos=True)
        done = False
        traj_len = 0
        while not done:
            a = env.get_opt_action()
            qs = env.get_q(discount)
            assert a in np.nonzero(np.max(qs) == qs)[0], 'not optimal!'
            np.testing.assert_allclose(env.get_v(discount), env.unwrapped.test_v(discount))
            np.testing.assert_allclose(env.get_v(discount), np.max(qs))
            _, r, done, _ = env.step(a)
            traj_len += 1
            assert r > -0.02, 'hit the wall somehow!' # 0.02 is bad...
        assert r >= env.success_reward, 'does not success!'
        assert traj_len >= env.unwrapped.min_dis, 'minimum length condition does not hold!'

def get_pick_env(l=16, task_length=2, num_obj_types=5):
    map_names = ['map{}'.format(i) for i in range(11, 21)]
    train_combos = list(product(range(10), range(10)))
    test_combos = train_combos
    env = PORGBEnv(
        PickGridWorld(
            map_names,
            train_combos=train_combos,
            test_combos=test_combos,
            num_obj_types=num_obj_types,
            task_length=task_length,
            window=4,
            seed=0,
        ),
        l=l,
    )
    return env

class TestPickEnv(unittest.TestCase):
    @ipdb_on_exception
    def try_env_config(self):
        for i in range(2, 12):
            with open('env_configs/pick/map0-inv-1-{}-min_dis-4'.format(i), 'rb') as f:
                env_config = dill.load(f)
            env = PORGBEnv(PickGridWorld(**env_config, task_length=1), l=16)
            control(env)
            env.render()

    @ipdb_on_exception
    def test_testcase(self):
        env = get_pick_env(l=3)
        filename = 'records/pick_records.pkl'
        if os.path.exists(filename):
            test_trajs(env, filename, index=(0, 0))
        else:
            record_trajs(env, filename, index=(0, 0))

    @ipdb_on_exception
    def control(self):
        env = PORGBEnv(
            PickGridWorld(
                map_names=['map0'],
                num_obj_types=3,
                task_length=1,
                train_combos=[(0, 0), (0, 1)],
                test_combos=[(0, 2)],
                obj_pos=[[(1, 1), (2, 2), (3, 3)]],
            ),
            l=24,
        )
        control(env)
        env.render()

    @ipdb_on_exception
    def obj_pos(self):
        env = PORGBEnv(
            PickGridWorld(
                map_names=['map0'],
                num_obj_types=3,
                task_length=1,
                train_combos=[(0, 0), (0, 1)],
                test_combos=[(0, 2)],
                obj_pos=[[(1, 1), (2, 2), (3, 3)]],
            ),
            l=16,
        )
        control(env)
        env.render()

    @ipdb_on_exception
    def test_planner(self): # may not be a good practice to mix all test together
        discount = 0.99
        env = get_pick_env()
        test_planner(env, discount)

if __name__ == "__main__":
    set_seed(1)
    unittest.main()
