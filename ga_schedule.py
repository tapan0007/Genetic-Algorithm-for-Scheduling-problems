"""
Implementation of a modified genetic algorithm for scheduling problems. In our case, we are solving warp s
cheduling problem in GPU's.
"""

import os
import sys
import numpy as np
import random

SCHEDULER = ["GTO", "LRR", "Youngest_Warp", "Youngest_Barrier_Warp", "Youngest_Finish_Warp"] 

class GeneticAlgorithm(object):

    def __init__(self, genetics):
        self.genetics = genetics

    def generate_intial_population(self):
     return [ [np.random.randint(0, len(SCHEDULER)) for chromo in chromosome ] for chromosome in self.genetics ]

    def random_selection(fitnesses):
    """
    This should a chromsome picked with the probability proportional to it's fitness
    """
    temp = [max(fitnesses) - fitness for fitness in fitnesses]
    total = sum(temp)
    temp = [(fitness * 1.0 / temp) for fitness in temp]
    selectrand = np.random.uniform(0, total)
    sel = 0
    index = 0
    for fitness in temp:
        sel += fitness
        if sel >= selectrand:
            return index
        index += 1
    return index

    def crossover():


    def mutation(offspring):


    def extend_chromosome():


    def convert_to_scheduler(chromosome):
        return  [SCHEDULER[index] for index in chromosome]

    def run():


class chromosome():
     def __init__(self, target,  popsize, size, prob_mutation, max_iterations)
        self.popsize = popsize
        self.size = size
        self.prob_mutation = prob_mutation
        self.max_iterations = max_iterations
        self.target = [ [0 for i in range(0, size )] for number in popsize  ]

    def find_fitness(index);
        """
        Open  a file and run the scheduler corrosponding to a specific chromosome
        """
        fitness = 0
        return fitness