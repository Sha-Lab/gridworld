import unittest
from itertools import product
import readchar
from env import GridWorld, ComboEnv, PORGBEnv

CONTROL_MAP = dict(w=0,s=1,a=2,d=3,e=4)

def control(env):
    env.reset((0, 0), sample_pos=False)
    done = False
    discount = 0.99
    while not done:
        env.render()
        c = readchar.readchar()
        if c == 'q':
            break
        elif c == 'r':
            env.reset(sample_pos=True, train=True)
        elif c == 'o':
            _, _, done, _ = env.step(env.get_random_opt_action(discount))
        elif c not in CONTROL_MAP:
            continue
        else:
            qs = env.get_qs(discount)
            a = CONTROL_MAP[c]
            o, r, done, info = env.step(a)
            diff = abs(r + discount * env.get_q(discount) - qs[a])
            assert diff < 1e-7, 'qs is not working! diff: {}'.format(diff)

class TestEnv(unittest.TestCase):
    def control(self):
        map_names = ['map{}'.format(i) for i in range(1, 21)]
        train_combos = list(product(range(10), range(10)))
        test_combos = train_combos
        env = PORGBEnv(ComboEnv(GridWorld(map_names, num_obj_types=5, train_combos=train_combos, test_combos=test_combos, window=1, seed=0)))
        control(env)
        env.render()

if __name__ == "__main__":
    unittest.main()
