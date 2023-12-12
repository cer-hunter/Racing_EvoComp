import numpy as np
import json
from operator import attrgetter

import car_racing_edited
import pygame
from utils import *

def keyboard_input():
    global quit, restart
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0

        if event.type == pygame.QUIT:
            quit = True

def set_tile_actions(tile_actions, tile, a):
    tile_actions[tile*2] = a[0]
    tile_actions[tile*2+1] = a[1] if a[1] > 0 else -a[2]
    return tile_actions

# Initialize track
env= car_racing_edited.CarRacing(render_mode="human")
env_noviz = car_racing_edited.CarRacing()
env_noviz.reset()
track_length = 140
print(f"Track Length: {track_length} Tiles")
tile_actions = np.zeros(track_length*2)

quit = False
env.reset()
total_reward = 0.0
steps = 0
restart = False
a = np.array([0.0, 0.0, 0.0])
while True:
    keyboard_input()
    s, r, terminated, truncated, info = env.step(a)
    tile_actions = set_tile_actions(tile_actions, info["current_tile"], a)
    total_reward -= r
    if steps % 200 == 0 or terminated or truncated:
        print("\naction " + str([f"{x:+0.2f}" for x in a]))
        print(f"step {steps} total_reward {total_reward:+0.2f}")
        print(info)
    steps += 1
    if terminated or truncated or restart or quit or info["current_tile"] >= track_length - 1:
        break
env.close()
training_history = {}
training_history[0] = {'fitness': total_reward, 'actions': tile_actions.tolist()}

with open("data/tile_training_history.json", "w") as f:
    json.dump(training_history, f, indent=4)