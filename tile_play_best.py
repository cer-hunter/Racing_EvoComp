import numpy as np
import json
from operator import attrgetter

import car_racing_edited
import pygame
from utils import *

def get_tile_action(tile_actions, tile):
    steer = np.clip(tile_actions[tile*2], -1, 1)
    accel = np.clip(tile_actions[tile*2+1], -0.8, 1)
    gas = accel if accel > 0 else 0
    brake = -accel if accel < 0 else 0
    return (np.array([steer, gas, brake]))

def get_tile_action_int(tile_actions, tile):
    steer = tile_actions[tile*2]
    accel = tile_actions[tile*2+1]
    gas = 1 if accel > 0.4 else 0
    brake = 0.8 if accel < -0.4 else 0
    return (np.array([steer, gas, brake]))

def evaluate(tile_actions, visualize=False):
    env = env_viz if visualize else env_noviz
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    a = get_tile_action_int(tile_actions, 0)
    while True:
        s, r, terminated, truncated, info = env.step(a)
        if info['speed'] < 0.1:
            tile_actions[info["current_tile"]*2+1] = 1
            tile_actions[info["current_tile"]*2-1] = 1            
            tile_actions[info["current_tile"]*2-3] = 1
        a = get_tile_action_int(tile_actions, info["current_tile"])
        total_reward -= r
        # if visualize and (steps % 200 == 0):
        #     print("\naction " + str([f"{x:+0.2f}" for x in a]))
        steps += 1
        # print(info["current_tile"])
        if terminated or truncated or restart or info['wheels_on_track'] == 0 or info["current_tile"] >= track_length -1:
            break
        if visualize:
            global quit
            quit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
            if quit:
                break
    return total_reward, tile_actions

with open("data/tile_training_history.json", "r") as f:
    training_history = json.load(f)
    prev_gens = max([int(x) for x in training_history.keys()]) + 1
    print(f"Starting from generation {prev_gens}, last best fitness: {training_history[str(prev_gens - 1)]['fitness']}")
tile_actions = np.array(training_history[f"{prev_gens - 1}"]['actions'])

env_viz = car_racing_edited.CarRacing(render_mode="human")
env_noviz = car_racing_edited.CarRacing()
env_noviz.reset()
track_length = 140

f, _ = evaluate(tile_actions, visualize=True)
print(f"Fitness: {f}")