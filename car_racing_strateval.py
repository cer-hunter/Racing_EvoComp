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

def equal(input1, input2):
    if input1 == input2:
        return 1
    else:
        return 0

def less(input1, input2):
    if input1 < input2:
        return 1
    else:
        return 0
def max(input1, input2):
    if input1>=input2:
        return input1
    else:
        return input2

#Memory Primitives

memory = np.zeros(6) #using global memory so that the array doesn't have to be passed as a variable by the tree

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
    
def intreturn(x):
    return(x)

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

obs_size = 6 # Car Racing has been edited to use #of tiles, pos x, pos y, hull angle and speed as the observations
pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float, float, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
#pset.addPrimitive(operator.mul, [float, float], float)
#pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(read, [int], float) 
pset.addPrimitive(write, [float, int], float)
pset.addPrimitive(if_then_else, [float, float, float], float)
pset.addPrimitive(equal, [float, float], float)
pset.addPrimitive(less, [float, float], float)
pset.addPrimitive(max, [float, float], float)
pset.addPrimitive(limit, [float, float, float], float)
pset.addPrimitive(intreturn, [int], int)
for i in range(0, memory.size):
   pset.addTerminal(i, int)
for i in range(0, 285):
   pset.addTerminal(i, float)


pset.renameArguments(ARG0="TileCount")
pset.renameArguments(ARG1="PosX")
pset.renameArguments(ARG2="PosY")
pset.renameArguments(ARG3="CarAngle")
pset.renameArguments(ARG4="Speed")
pset.renameArguments(ARG5="Wheels") #wheels on track


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

def discrete_wrapper(action): #defines a specific set of actions the car can take (basically discretizes them completely there is no continous spectrum)
    if action == 0: # if number is 0 do nothing
        return np.array([0, 0])
    elif action > 0 and action <= 1: #action in range 0-1 turn left and gas
        return np.array([-1, 1])
    elif action > 1 and action <=2: #action in range 1-2 turn right and gas
        return np.array([1, 1])
    elif action > 2 and action <=3: #action in range 2-3 brake
        return np.array([0, -0.8])
    else: #otherwise just gas
        return np.array([0,1])
        
    

# evaluates the fitness of an individual policy
def evalRL(policy, vizualize=False):
    env = env_viz if vizualize else env_noviz
    num_episode = 5
    # transform expression tree to functional Python code
    action = numpy.zeros(2)
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
             # use the expression tree to compute action from action_wrapper
            action[0] = numpy.clip(get_action(observation[0],observation[1],observation[2],observation[3],observation[4], observation[5]), -1, 1)
            action[1] = numpy.clip(get_action(observation[0],observation[1],observation[2],observation[3],observation[4], observation[5]), -1, 1)
            action = action_wrapper(action)
            # use the expression tree to compute action from discrete_wrapper
            #action = numpy.clip(get_action(observation[0],observation[1],observation[2],observation[3],observation[4], observation[5]), 0, 4)
            #action = discrete_wrapper(action)
            try:
                observation, reward, done, truncated, info = env.step(action)
            except:
                return (0,)
            episode_reward += reward
            num_steps += 1
        fitness += episode_reward
    return (fitness / num_episode,)

#changeable strategy...
strategy = gp.PrimitiveTree.from_string("", pset) #enter best strategy tree here
print(strategy)
evalRL(policy = strategy, vizualize=True)

#best strategy from action_wrapper
#write(sin(limit(sub(Speed, read(3)), sub(PosX, PosX), max(CarAngle, read(3)))), equal(max(sin(max(CarAngle, PosX)), write(if_then_else(CarAngle, write(if_then_else(PosX, add(PosX, Speed), PosX), equal(PosX, PosX)), sub(add(sub(PosX, PosX), limit(Speed, Speed, PosX)), add(PosX, Speed))), Wheels)), max(PosX, sub(read(equal(max(max(sub(max(CarAngle, Speed), add(PosX, Speed)), write(if_then_else(if_then_else(PosX, CarAngle, PosY), max(PosX, sub(Speed, PosX)), sin(sin(PosX))), Wheels)), read(equal(max(max(limit(sub(Speed, Speed), max(PosY, PosY), write(PosY, 5)), write(if_then_else(if_then_else(PosY, max(PosX, Speed), PosY), max(PosX, sub(Speed, PosX)), sin(PosX)), Wheels)), PosY), sin(PosX)))), PosX)), max(PosX, read(3))))))