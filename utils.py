import os
import sys
import random
import shutil
import pygame
import imageio
import inspect
import traceback
import contextlib
import numpy as np
from collections import defaultdict


def np_sample(l, random=None):
    if random is None:
        random = np.random
    return l[random.choice(len(l))]

# convert a list to two dictionary, index to element and element to index
# list will first pass through set to make element unique
def index_dict(l):
    l = list(enumerate(set(l)))
    i2e = dict(l)
    e2i = dict([(e, i) for i, e in l])
    return i2e, e2i

# create a dir if it does not exist
# rm: whether to first remove the original one if it exists
def mkdir(path, rm=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)

def rmf(path):
    t = get_filetype(path)
    if t == 0:
        os.remove(path)
    elif t == 1:
        os.rmdir(path)
    elif t == 2:
        shutil.rmtree(path)

# this is a decorator to a method or function, such that it called ipdb on exception
def ipdb_on_exception(origin_callable):
    if inspect.ismethod(origin_callable):
        def debug_method(self, *args, **kwargs):
            from ipdb import slaunch_ipdb_on_exception
            with slaunch_ipdb_on_exception():
                return origin_callable(self, *args, **kwargs)
        return debug_method
    else:
        def debug_func(*args, **kwargs):
            from ipdb import slaunch_ipdb_on_exception
            with slaunch_ipdb_on_exception():
                return origin_callable(*args, **kwargs)
        return debug_func

def set_seed(r, p=None):
    if p is None:
        p = r
    random.seed(r)
    np.random.seed(p)

def discount_cumsum(xs, discount=0.99):
    r = 0.0
    res = []
    for x in xs[::-1]:
        r = r * discount + x
        res.append(r)
    return res[::-1]

def four_directions(pos):
    x, y = pos
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def load_pygame_image(name):
    image = pygame.image.load(name)
    return image

class ImgSprite(pygame.sprite.Sprite):
    def __init__(self, rect_pos=(5, 5, 64, 64)):
        super(ImgSprite, self).__init__()
        self.image = None
        self.rect = pygame.Rect(*rect_pos)

    def update(self, image):
        if isinstance(image, str):
            self.image = load_pygame_image(image)
        else:
            self.image = pygame.surfarray.make_surface(image)

class Render(object):
    def __init__(self, size=(320, 320)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.group = pygame.sprite.Group(ImgSprite()) # the group of all sprites

    def render(self, img):
        img = np.asarray(img).transpose(1, 0, 2)
        self.group.update(img)
        self.group.draw(self.screen)
        pygame.display.flip()
        e = pygame.event.poll()

def chunk(l, n):
    sz = len(l)
    assert sz % n == 0, 'cannot be evenly chunked'
    for i in range(0, sz, n):
        yield l[i:i+n]

def save_gif(filename, imgs, duration=0.2):
    imageio.mimsave(filename, imgs, 'GIF', duration=duration)

@contextlib.contextmanager
def opens(filenames, *args, **kwargs):
    files = [open(filename, *args, **kwargs) if isinstance(filename, str) else open(filename, *args, **kwargs, closefd=False) for filename in filenames]
    try:
        # warning: make sure filenames does not change during this process
        yield files
    finally:
        for f, filename in zip(files, filenames):
            f.close()

def multi_print(*args, files=[sys.stdout], **kwargs):
    _ = [print(*args, file=x, **kwargs) for x in files]

storage_lists = defaultdict(lambda: defaultdict(list))
storage_values = defaultdict(dict)

# Be careful that when we put or push value, we don't copy them!
class Storage:
    def __init__(self, key='.'):
        self._key = key
        self._values = storage_values[self._key]
        self._lists = storage_lists[self._key]

    @property
    def key(self):
        return self._key

    def put(self, key, val):
        self._values[key] = val

    def push(self, key, val):
        self._lists[key].append(val)

    def get_val(self, key):
        assert key in self._values, 'key not found in value dictionary'
        return self._values[key]

    def get_list(self, key):
        assert key in self._lists, 'key not found in list dictionary'
        return self._lists[key]

    def get_values(self):
        return dict(self._values)

    def get_lists(self):
        return dict(self._lists)

    # used to clear data, after this
    def clear():
        storage_values[self.key] = dict()
        storage_lists[self.key] = defaultdict(list)
        self._values = storage_values[self.key]
        self._lists = storage_lists[self.key]

    def remove_value(key):
        del self._values[key]

    def remove_list(key):
        del self._lists[key]


# Currently it only outputs to files or stdout, a better version would be passed in a opener
class FileLogger:
    def __init__(self, files=[1], prefix='', output_configs=dict(mode='w')):
        self._files = files
        self._prefix = prefix
        self._output_configs = output_configs

    @property
    def files(self):
        return self._files

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        self._prefix = value

    @property
    def output_configs(self):
        return self._output_configs

    def new_files(files):
        return self.__class__(self.key, files, self.prefix)

    def add_file(f):
        return self.__class__(self.key, self.files + [f], self.prefix) 

    def info(self, *vals):
        if self.prefix != '':
            vals = ('[{}]'.format(self.prefix),) + vals
        with opens(self.files, **self.output_configs) as fs:
            multi_print(*vals, files=fs)

# return filename and get access to testfile that you can write and read
# provide cleaning up the test file
class TmpFile:
    def __init__(self, base_dir='.TmpFile', rm=True):
        self.base_dir = base_dir
        exist = os.path.exists(self.base_dir)
        self.rm = rm and (not exist)
        assert not exist, 'tmp base dir already exist!'
        mkdir(self.base_dir)

    def path(self, relative_path):
        return os.path.join(self.base_dir, relative_path)

    def __del__(self):
        if self.rm:
            rmf(self.base_dir)
