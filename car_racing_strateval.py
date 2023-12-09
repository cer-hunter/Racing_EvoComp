import math
import numpy 
import pygame
import random
import multiprocessing
import operator
from operator import attrgetter

import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from utils import *

import gymnasium as gym
import car_racing_edited

#this file is similar to the car racing controller except it's only function is to demonstrate strategies that have been found, not to generate new ones

#primitive definitions
def protectedDiv(left, right):
    try: return truncate(left, 8) / truncate(right, 8)
    except ZeroDivisionError: return 0
    
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

#Memory Primitives

memory = np.zeros(8) #using global memory so that the array doesn't have to be passed as a variable by the tree

def read(y):
    if y < memory.size:
        return memory[y]
    else:
        return 0

def write(x, y):
    if y < memory.size:
        old_mem = memory[y]
        memory[y] = x
        return old_mem
    else:
        return 0
    
def intreturn(x): #to add primitive that returns an int
    return x

def limit(input, minimum, maximum): #unused
    return min(max(input,minimum), maximum)

# helper function to limit decimal places
def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)
    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

obs_size = 5 # Car Racing has been edited to use #of tiles, pos x, pos y, hull angle and speed as the observations
pset = gp.PrimitiveSetTyped("MAIN", [int, float, float, float, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(read, [int], float) 
pset.addPrimitive(write, [float, int], float)
pset.addPrimitive(if_then_else, [float, float, float], float)
pset.addPrimitive(limit, [float, float, float], float)
pset.addPrimitive(intreturn, [int], int)
for i in range(0, memory.size):
   pset.addTerminal(i, int)

pset.renameArguments(ARG0="TileCount")
pset.renameArguments(ARG1="PosX")
pset.renameArguments(ARG2="PosY")
pset.renameArguments(ARG3="CarAngle")
pset.renameArguments(ARG4="Speed")


env_noviz = car_racing_edited.CarRacing()
env_viz = car_racing_edited.CarRacing(render_mode="human")

def action_wrapper(action): 
    #print(action)
    #for steering
    steer_action = action[0] 
    #print(steer_action)
    if steer_action > 1:
        steering = 1
    elif steer_action < -1:
        steering = -1
    else:
        steering = steer_action
    #for gas/brake
    gas_action = action[1]
    #print(gas_action)
    if gas_action > 1:
        gas = 1
    elif gas_action < -1:
        gas = -1
    else:
        gas = gas_action
    #return full action array
    return numpy.array([steering, gas])
        
    

# evaluates the fitness of an individual policy
def evalRL(policy, vizualize=False):
    env = env_viz if vizualize else env_noviz
    num_episode = 20
    # transform expression tree to functional Python code
    action = numpy.zeros(3)
    get_action = gp.compile(policy, pset) 
    fitness = 0
    for x in range(0, num_episode):
        done = False
        truncated = False
        # reset environment and get first observation
        observation = env.reset()
        observation = observation[0]
        episode_reward = 0
        num_steps = 0
        # evaluation episode
        while not (done or truncated):
            # use the expression tree to compute action
            action[0] = numpy.clip(get_action(observation[0],observation[1],observation[2],observation[3],observation[4]), -1, 1)
            action[1] = numpy.clip(get_action(observation[0],observation[1],observation[2],observation[3],observation[4]), -1, 1)
            action = action_wrapper(action)
            try:
                observation, reward, done, truncated, info = env.step(action)
            except:
                return (0,)
            episode_reward += reward
            num_steps += 1
        fitness += episode_reward
    return (fitness / num_episode,)

#changeable strategy... to reference arguments use pset.arguments[#]
strategy = gp.PrimitiveTree.from_string("read(intreturn(TileCount))", pset) #enter best strategy tree here
print(strategy)
evalRL(policy = strategy, vizualize=True)