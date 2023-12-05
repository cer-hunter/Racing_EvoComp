import math
import numpy as np
import pygame
import random
import multiprocessing
import operator
from operator import attrgetter

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import gymnasium as gym

import car_racing_edited
from utils import *

def get_tile_action(tile_actions, tile):
    return np.clip(tile_actions[tile*3:tile*3+3], -1.0, 1.0)

def evaluate(tile_actions, visualize=False):
    env = env_viz if visualize else env_noviz
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    a = get_tile_action(tile_actions, 0)
    while True:
        s, r, terminated, truncated, info = env.step(a)
        a = get_tile_action(tile_actions, info["current_tile"])
        total_reward -= r
        if visualize and (steps % 200 == 0 or terminated or truncated):
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
        steps += 1
        if terminated or truncated or restart or info['wheels_on_track'] == 0 or total_reward > 10:
            break
    return total_reward

def oneplus_lambda(x, fitness, gens=100, lam=20):
    x_best = x
    f_best = fitness(x)
    fits = np.zeros(gens)
    for g in range(gens):
        N = np.random.normal(size=(lam, len(x)))
        for i in range(lam):
            ind = x + N[i, :]
            f = fitness(ind)
            if f < f_best:
                f_best = f
                x_best = ind
        x = x_best
        fits[g] = f_best
        print(f_best)
    return fits, x_best



env_viz = car_racing_edited.CarRacing(render_mode="human")
env_noviz = car_racing_edited.CarRacing()
env_noviz.reset()
track_length = len(env_noviz.track)
print(f"Track Length: {track_length} Tiles")

# Give each tile a random set of actions (steering, acceleration, braking)
tile_actions = np.random.rand(track_length*3)
for i in range(track_length):
    tile_actions[i*3] = tile_actions[i*3] * 2 - 1

f = lambda x : evaluate(x)
fits, x_best = oneplus_lambda(tile_actions, f)


evaluate(tile_actions, visualize=True)
