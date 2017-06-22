import os
import sys
import numpy as np
import random
import subprocess

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
        This should a chromosome picked with the probability proportional to its fitness. Roulette
        Selection
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

    def least_fit(self, fitnesses):
        """
        The least fit will be the scheduler with maximum runtime
        """
        return fitnesses.index(max(fitnesses))

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

    def run(self):
        population = generate_intial_population()
        generation_num = 0
        while generation_num < self.max_iterations:
            print ("\n\n*************************************\n\n")
            print("Generation " + str(generation_num) + "\n")
            print ("\n\n*************************************\n\n")
            fitnesses = []
            index = 0
            for chromosome in population:     # Calculate the fitness of each chromosome
                print("Calculating fitness for chromosome index " + str(index))
                fitness_chromo = self.genetics.find_fitness(chromosome, index, generation_num, 1)
                print("The fitness of the chromosome with index " + str(fitness))
                fitnesses.append(fitness_chromo)

            print("The fitnesses of the chromosome in generation " + generation_num + " are\n")
            print(fitnesses)
            print("Best fitness among this is : " + min(fitnesses))

            new_chromosomes, new_fitnesses = [], []

            for k in range(0, int(0.85 * self.genetics.popsize) - 1):
                parent1 = random_selection(fitnesses)
                parent2 = parent1
                while parent1 != parent2:
                    parent2 = random_selection(fitnesses)
                off1, off2 = crossover(parent1, parent2)
                mutation(off1)
                mutation(off2)
                new_chromosomes.append(off1)
                new_chromosomes.append(off2)

            for chromo in new_chromosomes:
                new_fitnesses.append(find_fitness(chromo, "temp", "temp", 0))

            chromosome = chromosome + new_chromosomes
            fitnesses = fitnesses + new_fitnesses
            for k in range(0, int(0.85 * self.genetics.popsize) -  1):
                leastfit1 = least_fit(fitnesses)
                del chromosome[leastfit1]
                def fitnesses[leastfil1]
                leastfit2 = least_fit(fitnesses)
                del chromosome[leastfit2]
                del fitnesses[leastfit2]
        return

class chromosome():
     def __init__(self,  popsize, size, prob_mutation, max_iterations)
        self.popsize = popsize
        self.size = size
        self.prob_mutation = prob_mutation
        self.max_iterations = max_iterations
        self.target = [ [0 for i in range(0, size )] for number in popsize  ]
        dict_fitnesses = {}

    def find_fitness(chromosome, index, generation, savelog);
        """
        You can your own custom fitness function
        """
        if chromosome in dict_fitnesses:
            return dict_fitness[chromosome]
        fitness = 0
        file1= open('data/' + "generation_" + generation + "/" + "data_" + index, "w")
        for sch in chromosome:
            file1.write(str(sch) + ' ')
        file1.close()
        os.system("python script.py")
        file1 = open('data/' + "generation_" + generation + "/" + "data_" + index, "r")
        fitness = int(file1.readlines[0].strip(' \n'))
        file1.close()
        dict_fitness[chromosome] = fitness
        if savelog == 0:
            return fitness
        file2 = open('data/' + "generation_" + generation + "/" + "fitness_" + index, "w")
        file2.write(str(fitness))
        file2.close()
        return fitness

if __name__ == "__main__":
    GeneticAlgorithm(chromosome(100, 8, 0.1, 100))
    return
