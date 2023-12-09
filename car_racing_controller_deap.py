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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalRL)
toolbox.register("select", tools.selDoubleTournament, fitness_size = 3, parsimony_size = 1, fitness_first = True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(42)
# set to the number of cpu cores available
num_parallel_evals = 20 #change based on CPU host

population_size = 24 #can be tweaked for better results
num_generations = 50 
prob_xover = 0.9
prob_mutate = 0.3

pop = toolbox.population(n=population_size)

# HallOfFame archives the best individuals found so far,
# even if they are deleted from the population.
hof = tools.HallOfFame(1)  # We keep the single best.

# configues what stats we want to track
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

#setup parallel evaluations
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=num_parallel_evals)
    toolbox.register("map", pool.map)

    # run the evolutionary algorithm
    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        prob_xover,
        prob_mutate,
        num_generations,
        stats=mstats,
        halloffame=hof,
        verbose=True
    )

    pool.close()
    pool.join()

    best_fits = log.chapters["fitness"].select("max")
    best_fits_size = log.chapters["size"].select("avg")
    best_fit = truncate(hof[0].fitness.values[0], 0)

    print("Best fitness: " + str(best_fit))
    print(hof[0])

    evalRL(policy=hof[0], vizualize=True)

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(best_fits, "b-", label="Best Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(best_fits_size, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.savefig("fitness.png")

    nodes, edges, labels = gp.graph(hof[0])
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.png")  # write tree graph to PNG file
    
