import numpy as np
from operator import attrgetter

import car_racing_edited
from utils import *

def init_tile_actions(track_length):
    return np.random.rand(track_length*2) * 2 - 1

def get_tile_action(tile_actions, tile):
    steer = tile_actions[tile*2]
    accel = tile_actions[tile*2+1]
    gas = accel if accel > 0 else 0
    brake = -accel if accel < 0 else 0
    return np.array([steer, gas, brake])

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
        if visualize and (steps % 200 == 0):
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
        steps += 1
        if terminated or truncated or restart or info['wheels_on_track'] == 0 or total_reward > 10:
            break
    # print(f"Total Reward: {total_reward}")
    return total_reward

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

while True:
    evaluate(x_best, visualize=True)