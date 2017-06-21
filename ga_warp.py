import os
import sys
import numpy as np
import random

"""
TODO: Implement Bayesian Optimisation Algorithm
"""

SCHEDULER = ["GTO", "LRR", "Youngest_Warp", "Youngest_Barrier_Warp", "Youngest_Finish_Warp"]

class GeneticAlgorithm(object):

    def __init__(self, genetics):
        self.genetics = genetics

    def generate_intial_population(self):
     return [ [np.random.randint(0, len(SCHEDULER)) for chromo in chromosome ] for chromosome in self.genetics.target ]

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

    def crossover(parent1. parent2):
        """
        Produces 2 offspring as a result of crossover.
        """
        randindex = np.random.randint(1, size)
        return parent1[:randindex] + parent2[randindex:], parent2[:randindex] + parent1[randindex:]


    def mutation(self, offspring):
        """
        """
        for i in range(0, len(offspring)):
            if self.prob_crossover > np.random.uniform():
                value = np.random.randint(0, self.size)
                offspring[i] = index
        return
    def extend_chromosome(self, chromosome):
        """
        If the size of the chromosomes can vary, add some random scheduler at the end to make
        the chromosome valid
       """
        chromosome.append(0)
        return

    def convert_to_scheduler(chromosome):
        return  [SCHEDULER[index] for index in chromosome]

    def run():


class chromosome():
     def __init__(self, target,  popsize, size,prob_crossover, prob_mutation, max_iterations)
        self.popsize = popsize
        self.size = size
        sel.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.max_iterations = max_iterations
        self.target = [ [0 for i in range(0, size )] for number in popsize  ]

    def find_fitness(chromosome, index, generation);
        """
        You can your own custom fitness function
        """
        fitness = 0
        file = open('data/' + "generation_" + generation + "/" + "data_" + index, "w")
        for sch in chromosome:
            file.write(str(sch) + ' ')
        file.close()


        return fitness


if __name__ == "__main__":
    return




