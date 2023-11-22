import math
import numpy 
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

#area for primitives definitioins and such (to be added later)

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

obs_size = 3 # CartPole-v1 has 4 variables in each observation (change for other tasks)
pset = gp.PrimitiveSet("MAIN", obs_size) 

env_noviz = gym.make("CartPole-v1")
env_viz = gym.make("CartPole-v1", render_mode="human")

def action_wrapper(action): #placeholder for action wrapper
    return 0

# evaluates the fitness of an individual policy
def evalRL(policy, vizualize=False):
    env = env_viz if vizualize else env_noviz
    num_episode = 20
    # transform expression tree to functional Python code
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
            action = get_action(observation[0], observation[1], observation[2], observation[3]) #change for number of observations
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
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(42)
# set to the number of cpu cores available
num_parallel_evals = 4

population_size = 24
num_generations = 50
prob_xover = 0.9
prob_mutate = 0.1

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

best_fits = log.chapters["fitness"].select("max")
best_fit = truncate(hof[0].fitness.values[0], 0)

print("Best fitness: " + str(best_fit))
print(hof[0])

evalRL(policy=hof[0], vizualize=True)
