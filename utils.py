import pygame
import random
import numpy as np


def four_directions(pos):
    x, y = pos
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def discount_cumsum(xs, discount=0.99):
    r = 0.0
    res = []
    for x in xs[::-1]:
        r = r * discount + x
        res.append(r)
    return res[::-1]

def chunk(l, n):
    sz = len(l)
    assert sz % n == 0, 'cannot be evenly chunked'
    for i in range(0, sz, n):
        yield l[i:i+n]

def extract(d, *args):
    ret = []
    for k in args:
        ret.append(d[k])
    return ret

def set_seed(r, p=None):
    if p is None:
        p = r
    random.seed(r)
    np.random.seed(p)

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

