import os
import sys
import numpy as np
import random
import subprocess

NORMALIZE = 500
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
        temp = [max(fitnesses) - fitness + NORMALIZE for fitness in fitnesses] # Every chromosome should have non zero probability
        newfitness = temp
        total = sum(temp)
        temp = [(fitness * 1.0 / total) for fitness in temp]
        selectrand = np.random.uniform(0, total)
        #print(temp)
        #print(fitnesses)
        #print(selectrand)
        sel = 0
        index = 0
        for fitness in newfitness:
            sel += fitness
            if sel >= selectrand:
                return index
            index += 1
        return index

    def maintain_diversity(self, parent , off):
        mismatch = 0
        for i in range(0, len(parent)):
            if parent[i] != off[i]:
                mismatch = mismatch + 1
        if mismatch > 2:
            return True
        return False

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
            if self.genetics.prob_mutation > np.random.uniform():
                value = np.random.randint(0, 5)
                offspring[i] = value
        return offspring

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
        fitnesses = []
        while generation_num < self.genetics.max_iterations:
            generation_num = generation_num +  1
            print ("\n*************************************\n")
            print("Generation " + str(generation_num) + "\n")
            print ("*************************************\n")
            fitnesses = []
            index = 1
            new_chromo_fit = []
            for chromo in population:     # Calculate the fitness of each chromosome
                fitness_chromo = self.genetics.find_fitness(chromo, index, generation_num, 1)
                fitnesses.append(fitness_chromo)
                new_chromo_fit.append((fitness_chromo, chromo))
                index = index + 1

            print("Chromosomes in generation " + str(generation_num)  + " : " + str(population) + "\n")
            print("Chromosomes fitness in generation " + str(generation_num) + " : " + str(fitnesses) + "\n")
            print("Best fitness among this is : " + str(min(fitnesses)) + "\n")

            new_chromosomes, new_fitnesses = [], []
            print("Performing mutation and crossover\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            for k in range(0,  int(0.9 * self.genetics.popsize)):
                parent1 = self.random_selection(fitnesses)
                parent2 = parent1
                iterations = 0
                while parent1 == parent2 and iterations < 100:
                    parent2 = self.random_selection(fitnesses)
                    iterations = iterations + 1
                #print(str(parent1) + " " + str(parent2))
                off1, off2 = parent1, parent2
                if np.random.uniform() > self.genetics.prob_crossover:
                    continue
                else:
                    off1, off2 = self.crossover(population[parent1], population[parent2])
                self.mutation(off1)
                self.mutation(off2)
                if self.maintain_diversity(population[parent1], off1) == True:
                    new_chromosomes.append(off1)
                if self.maintain_diversity(population[parent2], off2) == True:
                    new_chromosomes.append(off2)

            for chromo in new_chromosomes:
                fitness_ch = self.genetics.find_fitness(chromo, 200, 200, 0)
                new_chromo_fit.append((fitness_ch, chromo))
                new_fitnesses.append(fitness_ch)

            new_chromo_fit.sort()
            population = [f[1] for f in new_chromo_fit[:self.genetics.popsize]]
            print("Newly constructed chromosome " + str(new_chromosomes))
            print("Newly constructed chromosomes fitness" + str(new_fitnesses))
            #print("Best fitness among the newly created Chromosome is : " + str(min(new_fitnesses)))
            continue
            """
            chromosome = population + new_chromosomes
            fitnesses = fitnesses + new_fitnesses
            assert len(chromosome) == len(fitnesses)
            #print("Newly constructed chromosomes "  + str(chromosome))
            #print("Newly constructed chromosomes fitnesses " + str(fitnesses))
            for k in range(0, int(0.9 * self.genetics.popsize)):
                leastfit1 = self.least_fit(fitnesses)
                del chromosome[leastfit1]
                del fitnesses[leastfit1]
                leastfit2 = self.least_fit(fitnesses)
                del chromosome[leastfit2]
                del fitnesses[leastfit2]
            population = chromosome
            #print("Chromosomes left after crossovers and mutations " + str(chromosome))
            #print("Chromosomes fitness after crossovers and mutations " + str(fitnesses))
            """
        print("The best fitness solution found is " + str(min(fitnesses)) + " " + str(fitnesses.index(min(fitnesses))))
        return

class chromosome():
    def __init__(self,  popsize, size, prob_mutation, prob_crossover, max_iterations, dict_fitness):
        self.popsize = popsize
        self.size = size
        self.prob_mutation = prob_mutation
        self.prob_crossover = prob_crossover
        self.max_iterations = max_iterations
        self.target = [ [0 for i in range(0, size )] for number in range(0, popsize)  ]
        self.dict_fitness = dict_fitness

    def find_fitness(self, chromo, index, generation, savelog):
        """
        You can your own custom fitness function
        """
        if tuple(chromo) in self.dict_fitness.keys():
            return self.dict_fitness[tuple(chromo)]
        fitness = 0
        file1 = open('data/' + "generation_" + str(generation) + "/" + "data_" + str(index) + ".txt", "w")
        for sch in chromo:
            file1.write(str(sch) + ' ')
        file1.close()
        os.system("python script.py " + str(generation) + " " + str(index) + " >> pythonlog.txt")
        print("###################\n")
        file1 = open('data/' + "generation_" + str(generation) + "/" + "fitness_" + str(index) + ".txt", "r")
        fitness = int(file1.readlines()[0])
        self.dict_fitness[tuple(chromo)] = fitness
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
    #print(chromosome(1,3,0.1,1, {}).find_fitness([0, 1, 0], 200, 200, 1))
    GeneticAlgorithm(chromosome(40, 13, 0.4, 0.9, 30, {})).run()
