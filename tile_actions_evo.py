import numpy as np
import json
from operator import attrgetter

import car_racing_edited
import pygame
from utils import *

def init_tile_actions(track_length):
    return np.random.rand(track_length*2) * 2 - 1
    # return np.tile([0.0, 1.0], track_length)

def get_tile_action(tile_actions, tile):
    steer = np.clip(tile_actions[tile*2], -1, 1)
    accel = np.clip(tile_actions[tile*2+1], -1, 1)
    gas = accel if accel > 0 else 0
    brake = -accel if accel < 0 else 0
    return (np.array([steer, gas, brake]))

def get_tile_action_int(tile_actions, tile):
    steer = np.round(tile_actions[tile*2]).astype(int)
    accel = tile_actions[tile*2+1]
    gas = 1 if accel > 0.5 else 0
    brake = 1 if accel < -0.5 else 0
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
        a = get_tile_action_int(tile_actions, info["current_tile"])
        total_reward -= r
        # if visualize and (steps % 200 == 0):
        #     print("\naction " + str([f"{x:+0.2f}" for x in a]))
        steps += 1
        if terminated or truncated or restart or info['wheels_on_track'] == 0 or (a[1] == 0.0 and info['speed'] < 0.1):
            break
        if visualize:
            global quit
            quit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
            if quit:
                break
    return total_reward

# Initialize track
env_viz = car_racing_edited.CarRacing(render_mode="human")
env_noviz = car_racing_edited.CarRacing()
env_noviz.reset()
track_length = len(env_noviz.track)
print(f"Track Length: {track_length} Tiles")

# Read previous training history or initialize new
try:
    with open("tile_training_history.json", "r") as f:
        training_history = json.load(f)
        prev_gens = max([int(x) for x in training_history.keys()]) + 1
        print(f"Starting from generation {prev_gens}, last best fitness: {training_history[str(prev_gens - 1)]['fitness']}")
except:
    training_history = {}
    prev_gens = 0

if prev_gens == 0:
    tile_actions = init_tile_actions(track_length)
else:
    tile_actions = np.array(training_history[f"{prev_gens - 1}"]['actions'])

# Run training
f = lambda x : evaluate(x)
# fits, actions_best, x_history = oneplus_lambda(x=tile_actions, fitness=f, gens=100)
fits, actions_best, x_history = cma_es(x=tile_actions, fitness=f, gens=100, popsize=100)



# Save training history
for g in x_history.keys():
    training_history[prev_gens + g] = x_history[g]

with open("tile_training_history.json", "w") as f:
    json.dump(training_history, f, indent=4)

# Visualize best action sequence
global quit
quit = False
while True and not quit:
    evaluate(actions_best, visualize=True)