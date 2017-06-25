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

    def random_selection(self, fitnesses):
        """
        This should a chromosome picked with the probability proportional to its fitness. Roulette
        Selection
        """
        temp = [max(fitnesses) - fitness for fitness in fitnesses]
        total = sum(temp)
        temp = [(fitness * 1.0 / total) for fitness in temp]
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

    def crossover(self, parent1, parent2):
        """
        Produces 2 offspring as a result of crossover.
        """
        randindex = np.random.randint(1, self.genetics.size)
        return parent1[:randindex] + parent2[randindex:], parent2[:randindex] + parent1[randindex:]

    def mutation(self, offspring):
        """
        """
        for i in range(0, len(offspring)):
            if self.prob_crossover > np.random.uniform():
                value = np.random.randint(0, self.genetics.size)
                offspring[i] = index
        return

    def extend_chromosome(self, chromosome):
        """
        If the size of the chromosomes can vary, add some random scheduler at the end to make
        the chromosome valid
        """
        chromosome.append(0)
        return

    def convert_to_scheduler(self, chromosome):
        return  [SCHEDULER[index] for index in chromosome]

    def run(self):
        population = self.generate_intial_population()
        self.target = population
        print("Initial Population")
        print(population)
        generation_num = 0
        while generation_num < self.genetics.max_iterations:
            generation_num = generation_num +  1
            print ("\n*************************************\n")
            print("Generation " + str(generation_num) + "\n")
            print ("*************************************\n")
            fitnesses = []
            index = 1
            for chromosome in population:     # Calculate the fitness of each chromosome
                print("Calculating fitness for chromosome index " + str(index))
                fitness_chromo = self.genetics.find_fitness(chromosome, index, generation_num, 1)
                print("The fitness of the chromosome with index " + str(fitness_chromo))
                fitnesses.append(fitness_chromo)
                index = index + 1

            print("The fitnesses of the chromosome in generation " + str(generation_num) + " are:")
            print(fitnesses)
            print("Best fitness among this is : " + str(min(fitnesses)))

            new_chromosomes, new_fitnesses = [], []
            print("Perform mutation and crossover")
            for k in range(0, int(0.85 * self.genetics.popsize)):
                parent1 = self.random_selection(fitnesses)
                parent2 = parent1
                while parent1 != parent2:
                    parent2 = self.random_selection(fitnesses)
                off1, off2 = self.crossover(fitnesses[parent1], fitnesses[parent2])
                self.mutation(off1)
                self.mutation(off2)
                new_chromosomes.append(off1)
                new_chromosomes.append(off2)

            for chromo in new_chromosomes:
                new_fitnesses.append(find_fitness(chromo, "temp", "temp", 0))

            chromosome = chromosome + new_chromosomes
            fitnesses = fitnesses + new_fitnesses
            for k in range(0, int(0.85 * self.genetics.popsize)):
                leastfit1 = least_fit(fitnesses)
                del chromosome[leastfit1]
                del fitnesses[leastfit1]
                leastfit2 = least_fit(fitnesses)
                del chromosome[leastfit2]
                del fitnesses[leastfit2]
        return

class chromosome():
    def __init__(self,  popsize, size, prob_mutation, max_iterations, dict_fitness):
        self.popsize = popsize
        self.size = size
        self.prob_mutation = prob_mutation
        self.max_iterations = max_iterations
        self.target = [ [0 for i in range(0, size )] for number in range(0, popsize)  ]
        self.dict_fitness = dict_fitness

    def find_fitness(self, chromo, index, generation, savelog):
        """
        You can your own custom fitness function
        """
        if chromo in self.dict_fitness.keys():
            return self.dict_fitness[chromo]
        fitness = 0
        file1 = open('data/' + "generation_" + str(generation) + "/" + "data_" + str(index) + ".txt", "w")
        for sch in chromo:
            file1.write(str(sch) + ' ')
        file1.close()
        os.system("python script.py " + str(generation) + " " + str(index))
        file1 = open('data/' + "generation_" + str(generation) + "/" + "fitness_" + str(index) + ".txt", "r")
        fitness = int(file1.readlines()[0])
        """
        fitness = int(file1.readlines()[0])
        file1.close()
        self.dict_fitness[chromo] = fitness
        if savelog == 0:
            return fitness
        file2 = open('data/' + "generation_" + generation + "/" + "fitness_" + index, "w")
        file2.write(str(fitness))
        file2.close()
        """
        return fitness

if __name__ == "__main__":
    """
    The following is for demo. You can change the attributes for the genetic algorithm.
    """
    #print(chromosome(1,3,0.1,1, {}).find_fitness([0, 1, 0], 1, 1, 1))
    GeneticAlgorithm(chromosome(2, 3, 0.1, 3, {})).run()
