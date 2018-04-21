import numpy as np
from abc import ABC, abstractmethod


''' The problems '''
def get_problem_1():
    process_200 = np.random.randint(10, 1001, 200)
    process_100 = np.random.randint(100, 301, 100)
    processing = np.zeros(300)
    processing[:200] = process_200
    processing[200:] = process_100
    return processing

def get_problem_2():
    process_150_1 = np.random.randint(10, 1001, 150)
    process_150_2 = np.random.randint(400, 701, 150)
    processing = np.zeros(300)
    processing[:200] = process_150_1
    processing[200:] = process_150_2
    return processing

def get_problem_3():
    processing = [50, 50, 50]
    for i in range(100 - 51):
        processing.append(i + 51)
        processing.append(i + 51)
    return processing


''' Modules '''
class Initializer(ABC):
    def __init__(self, num_jobs, num_machines, population_size):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.population_size = population_size

    @abstractmethod
    def initialize(self):
        pass

class Equal_Initializer(Initializer):

    def initialize(self):
        population_init = []
        for i in range(self.population_size):
            per_machine = self.num_jobs // self.num_machines
            init = []
            for i in range(self.num_machines):
                init.append([i+1] * per_machine)
            init = np.ravel(np.asarray(init))
            if len(init) != self.num_jobs:
                for i in range(self.num_jobs - len(init)):
                    init = np.append(init, 1)
            population_init.append(init)
        return population_init

class Random_Initializer(Initializer):

    def initialize(self):
        init = np.zeros((self.population_size, self.num_jobs))
        for i in range(self.population_size):
            init[i] = np.random.randint(1, self.num_machines + 1, self.num_jobs)
        return init


class Selector():
    def __init__(self, selection_size):
        self.size = selection_size

    def select(self, population, evaluation):
        pass

class Roulette_Wheel_Selector(Selector):
    def select(self, population, evaluation):
        selection_list = []
        # for how many elements we want to select
        for elem in range(self.size):
            sum_fittness = 0
            for fittness in evaluation:
                sum_fittness = sum_fittness + fittness
            # calculate cumulated probabilities
            probabilities = []
            last_prob = 0
            for fittness in evaluation:
                new_prob = fittness/sum_fittness + last_prob
                probabilities.append(new_prob)
                last_prob = new_prob
            # select a chromosome according to its probability
            rand = np.random.random()
            for index, probability in enumerate(probabilities):
                if probability < rand:
                    continue
                else:
                    selection_list.append(population[index])
                    break
        return selection_list


class Tournament_Selector(Selector):
    # TODO
    def select(self, population, evaluation):
        return max(candidates)


class Recombiner(ABC):
    """
    Recombines two parents into two new children with possibly different genes
    """

    def __init__(self, cross_probability=.5):
        """
        Inits the recombiner with a defined crossover probability, defaulting to .5
        """
        self.crossover_probability = cross_probability

    def crossover_random_num(self):
        """
        Generates a random r to check if crossover occurs
        """
        return np.random.random()

    def get_random_pair(self, chromosomes):
        """
        Pulls out two random chromosomes to crossover.  If the same pair is pulled, generate another one until
        different chromosomes are pulled.
        """
        pair = np.random.randint(0, len(chromosomes), 2)
        while pair[0] == pair[1]:
            pair[1] = np.random.randint(0, len(chromosomes))
        return pair

    @abstractmethod
    def recombine(self, chromosomes):
        pass

class One_Point_Crossover(Recombiner):

    def recombine(self, chromosomes):
        """
        One point crossover implementation
        """
        pair = self.get_random_pair(chromosomes)
        # Define the parents
        mom = chromosomes[pair[0]]
        dad = chromosomes[pair[1]]
        # See if they crossover
        cross_chance = self.crossover_random_num()
        if cross_chance < self.crossover_probability:
            # Cross over occurs, generate crossover point
            crossover_point = np.random.randint(1, len(chromosomes[1]))
            child1, child2 = [], []
            from_mom = True
            # Step through alleles
            for i in range(len(chromosomes[1])):
                if i == crossover_point:  # if hit crossover point, swap allele selection
                    from_mom = False
                if from_mom:
                    child1.append(mom[i])
                    child2.append(dad[i])
                else:
                    child1.append(dad[i])
                    child2.append(mom[i])
            return child1, child2
        else:  # if no crossover, return parents
            return mom, dad


class Two_Point_Crossover(Recombiner):
    def recombine(self, chromosomes):
        """
        Two point crossover implementation
        """
        pair = self.get_random_pair(chromosomes)
        # Define the parents
        mom = chromosomes[pair[0]]
        dad = chromosomes[pair[1]]
        # See if they crossover
        cross_chance = self.crossover_random_num()
        if cross_chance < self.crossover_probability:
            # Cross over occurs, generate crossover points
            crossover_point1 = np.random.randint(1, len(chromosomes[1]))
            crossover_point2 = np.random.randint(1, len(chromosomes[1]))
            if crossover_point2 < crossover_point1:  # make sure they are in order
                temp = crossover_point2
                crossover_point2 = crossover_point1
                crossover_point1 = temp
            while crossover_point1 == crossover_point2:  # make sure they are not the same
                crossover_point2 = np.random.randint(1, len(chromosomes[1]))
            child1, child2 = [], []
            from_mom = True
            for i in range(len(chromosomes[1])):
                if i == crossover_point1:  # if hit crossover point, swap allele selection
                    from_mom = False
                if i == crossover_point2:  # if hit crossover point, swap allele selection
                    from_mom = True
                if from_mom:
                    child1.append(mom[i])
                    child2.append(dad[i])
                else:
                    child1.append(dad[i])
                    child2.append(mom[i])
            return child1, child2
        else:  # if no crossover, return parents
            return mom, dad

        
class Mutator():
    
    def random(self, offspring, num_jobs):
        """
        Changes each allel of each chromosome with a certain probabilty p_m 
        """
        mutated_offspring = []
        # mutate all chromosomes of the offspring
        for a in range(len(offspring)):
            chromosome = offspring[a]
            # mutate all allels of a chromsome with probability p_m
            for i in range(len(chromosome)):
                # define p_m randomly
                probabilty_m = np.random.random()
                # threshold, for which an allel get changed
                if probabilty_m < 0.5:
                    mutation = np.random.randint(1, num_jobs)
                    # as long as the mutation has the same value than the allel initially had, generate new mutation
                    while chromosome[i] == mutation:
                        mutation = np.random.randint(1, num_jobs)
                    # substitute allel by its mutation    
                    chromosome[i] = mutation 
                else: 
                    chromosome[i] = chromosome[i]
            # generate the new offspring by collecting all mutated chromosomes
            mutated_offspring.append(chromosome)
        return mutated_offspring
    
    def lazy(self, offspring, num_jobs):
        """
        Changes only one allel of each chromosome with a certain probabilty p_m
        """
        mutated_offspring = []
        # mutate all chromosomes of the offspring
        for a in range(len(offspring)):
            chromosome = offspring[a]
            # define p_m randomly
            probabilty_m = np.random.random()
            # threshold, for which an allel get changed
            if probabilty_m < 0.5:
                # defines the allel that gets changed and the mutation itself randomly
                allel = np.random.randint(0, (len(chromosome)))
                mutation = np.random.randint(1, num_jobs)
                # as long as the mutation has the same value than the allel initially had, generate new mutation
                while chromosome[allel] == mutation:
                    mutation = np.random.randint(1, num_jobs)
                # substitute allel by its mutation
                chromosome[allel] = mutation
                # generate the new offspring by collecting all mutated chromosomes
                mutated_offspring.append(chromosome)
            else:
                # collect the unmutated chromosome in the new offspring
                mutated_offspring.append(chromosome)
        return mutated_offspring     


''' Main algorithm '''
class Genetic_Algorithm:

    def __init__(self, problem, initializer, selector, recombiner, mutator, replacer):
        self.problem = problem

        self.population = initializer.initialize()

        self.selector = selector
        self.recombiner = recombiner
        self.mutator = mutator
        self.replacer = replacer

    def evaluate_population(self):
        """ Evalutator """
        evaluation = []
        # iterates all chromosomes in population and evaluates them
        for chromosome in self.population:
            # create a list of the machines
            eval_machines = np.unique(self.population)
            # calculate total processing time for every machine
            for index, job in enumerate(chromosome):
                eval_machines[job-1] = eval_machines[job-1] + self.problem[index]
            # take max machine-time
            evaluation.append(max(eval_machines))
        self.evaluation = evaluation

    def run_episode(self):
        evaluation = self.evaluate_population()
        self.selection = self.selector.select(self.population, self.evaluation)
        self.recombination = self.recombiner.recombine(self.selection)
        # print(self.recombination) # not sure if it works yet
        # print(np.shape(self.recombination))
        self.mutation = self.mutator.mutate(self.recombination, num_jobs-1)
        # print(self.mutation)
        # TODO replacer.replace(self.population, self.mutation)

''' Hyperparameters '''
population_size = 10
num_jobs = 300
num_machines = 20
selection_size = 4


''' Main script '''
problem_1 = get_problem_1()
#problem_2 = get_problem_2() # TODO: throws error !
problem_3 = get_problem_3()

equal_initializer = Equal_Initializer(num_jobs,num_machines,population_size)
random_initializer = Random_Initializer(num_jobs,num_machines,population_size) # TODO: change float output to int
one_point_crossover = One_Point_Crossover()
two_point_crossover = Two_Point_Crossover()
lazy_mutator = Lazy_Mutator()
roulette_wheel_selector = Roulette_Wheel_Selector(selection_size)

ga_1 = Genetic_Algorithm(problem_1, equal_initializer, roulette_wheel_selector, one_point_crossover, lazy_mutator, 0)
ga_2 = Genetic_Algorithm(problem_1, random_initializer, roulette_wheel_selector, two_point_crossover, lazy_mutator, 0)

print(np.shape(ga_1.population))
ga_1.run_episode()


