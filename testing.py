import math
import numpy as np
import pygame
import random
import multiprocessing
import operator
from operator import attrgetter
#place holder for now i dont know if we will want to use deap or pymoo or whatever
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import gymnasium as gym

import car_racing_edited
from utils import *

env = car_racing_edited.CarRacing(render_mode="human")

def get_controll_action(strategy, info):
    brake_point = strategy['brake_point']         # defined in cy
    turn_in_point = strategy['turn_in_point']     # defined in cy
    accel_point = strategy['accel_point']         # defined in cx

    # Get the relative position of the car with respect to the corner entry
    cx, cy = to_corner_coord(info["pos_x"], info["pos_y"])
    exit_angle = to_exit_angle(info["angle"])

    # Actions
    a = np.array([0.0, 0.0, 0.0])

    if cy < brake_point:
        # full throttle before braking zone
        a[1] = 1.0
    elif cy < turn_in_point:
        # start braking
        a[2] = 0.8
    elif abs(exit_angle) > 0:
        # stop braking and start turn-in
        if accel_point < 0:
            a[0] = -1.0
        else:
            a[0] = +1.0

    if cx < accel_point < 0 or cx > accel_point > 0:
        # start accelerating out of corner
        a[1] = +1.0

        # angle correction to make the car go straight on exit
        gain = 0.8
        a[0] = np.clip(exit_angle * gain, -1.0, +1.0)
    
    return a


strategy = {
    'brake_point': -17.0,
    'turn_in_point': +2.0,
    'accel_point': -5.0
}

if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, terminated, truncated, info = env.step(a)
            a = get_controll_action(strategy, info)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                print(info)
                cx, cy = to_corner_coord(info["pos_x"], info["pos_y"])
                print(f"Rel Corner Entry Coord {cx:+0.2f}, {cy:+0.2f}")
                print(f"Rel Corner Entry Angle {to_entry_angle(info['angle']):+0.2f}")
                print(f"Rel Corner Exit Angle {to_exit_angle(info['angle']):+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
