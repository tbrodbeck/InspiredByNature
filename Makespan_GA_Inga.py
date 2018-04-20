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
    @abstractmethod
    def initialize(self):
        pass

class Equal_Initializer(Initializer):
    def __init__(self, num_jobs, num_machines, population_size):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.population_size = population_size
    def initialize(self):
        per_machine = self.num_jobs // self.num_machines
        init = []
        for i in range(self.num_machines):
            init.append([i+1] * per_machine)
        init = np.ravel(np.asarray(init))
        if len(init) != self.num_jobs:
            for i in range(self.num_jobs - len(init)):
                init = np.append(init, 1)
        return init

class Random_Initializer(Initializer):
    def __init__(self, num_jobs, num_machines, population_size):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.population_size = population_size
    def initialize(self):
        init = np.zeros((self.population_size, self.num_jobs))
        for i in range(self.population_size):
            init[i] = np.random.randint(1, self.num_machines + 1, self.num_jobs)
        return init


class Selector():
    def roulette(self):
        raise NotImplementedError
    def tournament(self, candidates):
        return max(candidates)


class Recombiner():
    """
    Recombines two parents into two new children with possibly different genes
    """

    def __init__(self, chromosomes, cross_probability=.5):
        """
        Inits the recombiner with a defined crossover probability, defaulting to .5
        """
        self.chromosomes = chromosomes
        self.crossover_probability = cross_probability

    def crossover_random_num(self):
        """
        Generates a random r to check if crossover occurs
        """
        return np.random.random()

    def get_random_pair(self):
        """
        Pulls out two random chromosomes to crossover.  If the same pair is pulled, generate another one until
        different chromosomes are pulled.
        """
        pair = np.random.randint(0, len(self.chromosomes), 2)
        while pair[0] == pair[1]:
            pair[1] = np.random.randint(0, len(self.chromosomes))
        return pair

    def one_point_crossover(self):
        """
        One point crossover implementation
        """
        pair = self.get_random_pair()
        # Define the parents
        mom = self.chromosomes[pair[0]]
        dad = self.chromosomes[pair[1]]
        # See if they crossover
        cross_chance = self.crossover_random_num()
        if cross_chance < self.crossover_probability:
            # Cross over occurs, generate crossover point
            crossover_point = np.random.randint(1, len(self.chromosomes[1]))
            child1, child2 = [], []
            from_mom = True
            # Step through alleles
            for i in range(len(self.chromosomes[1])):
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

    def two_point_crossover(self):
        """
        Two point crossover implementation
        """
        pair = self.get_random_pair()
        # Define the parents
        mom = self.chromosomes[pair[0]]
        dad = self.chromosomes[pair[1]]
        # See if they crossover
        cross_chance = self.crossover_random_num()
        if cross_chance < self.crossover_probability:
            # Cross over occurs, generate crossover points
            crossover_point1 = np.random.randint(1, len(self.chromosomes[1]))
            crossover_point2 = np.random.randint(1, len(self.chromosomes[1]))
            if crossover_point2 < crossover_point1:  # make sure they are in order
                temp = crossover_point2
                crossover_point2 = crossover_point1
                crossover_point1 = temp
            while crossover_point1 == crossover_point2:  # make sure they are not the same
                crossover_point2 = np.random.randint(1, len(self.chromosomes[1]))
            child1, child2 = [], []
            from_mom = True
            for i in range(len(self.chromosomes[1])):
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
    #to-do!! (Inga)
    def random(self, offspring, num_jobs):
        probabilty_m = np.random.random()
        if probabilty_m < 0.1:
            #randomly choses a position that gets mutated
            bit = np.random.randint(0, (len(offspring) - 1))
            mutation = np.random.randint(1, num_jobs)            
            while offspring[bit] == mutation:
                mutation = np.random.randint(1, num_jobs)                
            offspring[bit] = mutation            
            return offspring        
        else:
            return offspring
    
    def lazy(self, offspring, num_jobs):
        probabilty_m = np.random.random()
        if probabilty_m < 0.1:
            #randomly choses a position that gets mutated
            bit = np.random.randint(0, (len(offspring) - 1))
            mutation = np.random.randint(1, num_jobs)        
            while offspring[bit] == mutation:
                mutation = np.random.randint(1, num_jobs)                
                offspring[bit] = mutation            
            return offspring      
        else:
            return offspring                       

class Genetic_Algotihm:

    def __init__(self, initializer, selector, recombiner, mutator, replacer):
        self.initializer = initializer
        self.selector = selector
        self.recombiner = recombiner
        self.mutator = mutator
        self.replacer = replacer

        self.population = self.initializer.initialize()

    def evaluate_population(self):
        for chromosome in self.population:
            for index, job in enumerate(chromosome):


''' hyperparameters '''
population_size = 1
num_jobs = 300
num_machines = 20

''' main script '''
#equal_initializer = Equal_Initializer(num_jobs,num_machines,population_size)
random_initializer = Random_Initializer(num_jobs,num_machines,population_size)

#ga_equal = Genetic_Algotihm(equal_initializer, 0,0,0,0)
ga_random = Genetic_Algotihm(random_initializer,0,0,0,0)
#print(ga_equal.population)
print(ga_random.population)



