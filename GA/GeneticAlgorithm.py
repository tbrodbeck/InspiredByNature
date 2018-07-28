import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import copy


''' The problems '''
class Problem(ABC):
    def get_time(self):
        """
        Calculates processing time
        """
        pass

    def get_machine_num(self):
        pass

class Problem_1(Problem):
    def get_time(self):
        process_200 = np.random.randint(10, 1001, 200)
        process_100 = np.random.randint(100, 301, 100)
        processing = np.zeros(300)
        processing[:200] = process_200
        processing[200:] = process_100
        return processing

    def get_machine_num(self):
        return 20

class Problem_2(Problem):
    def get_time(self):
        process_150_1 = np.random.randint(10, 1001, 150)
        process_150_2 = np.random.randint(400, 701, 150)
        processing = np.zeros(300)
        processing[:150] = process_150_1
        processing[150:] = process_150_2
        return processing

    def get_machine_num(self):
        return 20

class Problem_3(Problem):
    def get_time(self):
        processing = [50, 50, 50]
        for i in range(100 - 51):
            processing.append(i + 51)
            processing.append(i + 51)
        return processing

    def get_machine_num(self):
        return 50


''' Modules '''
class Initializer(ABC):
    """
    Initializes Chromosomes
    """
    def __init__(self, population_size):
        self.population_size = population_size

    @abstractmethod
    def initialize(self, num_jobs, num_machines):
        pass

class Equal_Initializer(Initializer):

    def initialize(self, num_jobs, num_machines):
        '''
        Initialize all machines wth an equal number of jobs in order.  If they can't evenly distribute, then just add the first machine
        at the end until full
        :param num_jobs:
        :param num_machines:
        :return:
        '''
        population_init = []
        for i in range(self.population_size): # generate a certain amount of this arrays
            per_machine = num_jobs // num_machines  # how many jobs per machine
            init = []
            for i in range(num_machines):  # fill jobs with same machine per_machine time
                init.append([i+1] * per_machine)
            init = np.ravel(np.asarray(init))  # flatten the array
            if len(init) != num_jobs:  # check if we filled it to the same size as the number of jobs
                for i in range(num_jobs - len(init)):
                    init = np.append(init, 1)
            population_init.append(init)
        return population_init


class Random_Initializer(Initializer):

    def initialize(self, num_jobs, num_machines):
        '''
        Initialize each machine with random jobs
        :param num_jobs:
        :param num_machines:
        :return:
        '''
        init = []
        for i in range(self.population_size):  # make many random chromosomes
            init.append(np.random.randint(1, num_machines + 1, num_jobs))
        return init


class Selector():
    """
    Selects Candidates
    """
    def __init__(self, selection_size):
        self.size = selection_size

    @abstractmethod
    def select(self, population, evaluation):
        pass

class Roulette_Wheel_Selector(Selector):
    def select(self, population, evaluation):
        selection_list = []
        # we prefer evaluations with lower value
        maximum = np.max(evaluation)
        eval_new = maximum - evaluation

        sum_fittness = 0
        for fittness in eval_new:
            sum_fittness = sum_fittness + fittness
        # calculate cumulated probabilities
        probabilities = []
        last_prob = 0
        for fittness in eval_new:
            new_prob = fittness / sum_fittness + last_prob
            probabilities.append(new_prob)
            last_prob = new_prob

        # for how many elements we want to select
        for elem in range(self.size):
            rand = np.random.random()
            # select a chromosome according to its probability
            for index, probability in enumerate(probabilities):
                if probability < rand:
                    continue
                else:
                    selection_list.append(population[index])
                    break
        return selection_list


class Tournament_Selector(Selector):
    """
    Tournament should not be used together with delete_all_replacer
    """
    def select(self, population, evaluation, tournaments=8, competitors=2):
        # First of all we need deepcopies in order to not manipulate our current population
        population_copy = copy.deepcopy(population)
        evaluation_copy = copy.deepcopy(evaluation)

        winner = []

        # initiate tournaments
        for s in range(tournaments):
            candidates = []
            values = []
            bestValue = 0
            bestPos = 0

            # construct arrays containing information about the competitors of a certain tournament
            for c in range(competitors):
                pos = np.random.randint(0, len(population_copy))
                elem = population_copy.pop(pos)
                candidates.append(elem)
                elem = evaluation_copy.pop(pos)
                values.append(elem)

            # determine winner
            for i in range(len(values)):
                if evaluation_copy[i] > bestValue:
                    bestValue = values[i]
                    bestPos = i

            # remember winner
            elem = candidates.pop(bestPos)
            winner.append(elem)
        return winner


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
        return pair[0], pair[1]

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

        
class Mutator(ABC):
    """
    Makes small changes in single chromosomes
    """
    @abstractmethod
    def mutate(self, offspring, num_jobs):
        pass

class Random_Mutator(Mutator):
    def mutate(self, offspring, num_machines):
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
                if probabilty_m < 0.1:
                    mutation = np.random.randint(1, num_machines)
                    # as long as the mutation has the same value than the allel initially had, generate new mutation
                    while chromosome[i] == mutation:
                        mutation = np.random.randint(1, num_machines)
                    # substitute allel by its mutation    
                    chromosome[i] = mutation 
                else: 
                    chromosome[i] = chromosome[i]
            # generate the new offspring by collecting all mutated chromosomes
            mutated_offspring.append(chromosome)
        return mutated_offspring

class Lazy_Mutator(Mutator):
    def mutate(self, offspring, num_machines):
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
            if probabilty_m < 0.1:
                # defines the allel that gets changed and the mutation itself randomly
                allel = np.random.randint(0, (len(chromosome)))
                mutation = np.random.randint(1, num_machines)
                # as long as the mutation has the same value than the allel initially had, generate new mutation
                while chromosome[allel] == mutation:
                    mutation = np.random.randint(1, num_machines)
                # substitute allel by its mutation
                chromosome[allel] = mutation
                # generate the new offspring by collecting all mutated chromosomes
                mutated_offspring.append(chromosome)
            else:
                # collect the unmutated chromosome in the new offspring
                mutated_offspring.append(chromosome)
        return mutated_offspring

class Replacer(ABC):
    """
    Replaces population chromosomes with the offspring chromosomes
    """
    @abstractmethod
    def replace(self, population, offspring, evaluationP, evaluationO):
        pass

class Delete_All(Replacer):
    def replace(self, population, offspring, evaluationP, evaluationO):
        """
        Replaces all chromosomes of the population by the chromosomes of the offspring
        """
        if len(offspring) < len(population):
            print('ERROR: should be len(offspring) >= len(population)')
            return False

        self.newPop = []
        # use chromosomes of the offspring in random order for the new population
        for i in range(len(population)):
            self.newPop.append(offspring.pop(np.random.randint(0, len(offspring))))

        return self.newPop

class Steady_State(Replacer):

    def __init__(self, number=4):

        self.number = number
    """
    Replaces the n worst chromosomes of the population by the n best chromosomes of the offspring
    """
    def replace(self, population, offspring, evaluationP, evaluationO):
        # delete the n worst chromosomes of the population
        for i in range(self.number):
            maxPos = evaluationP.index(max(evaluationP))
            evaluationP.pop(maxPos)
            population.pop(maxPos)
        # add the n best chromosomes of the offspring to the population
        for j in range(self.number):
            minPos = evaluationO.index(min(evaluationO))
            evaluationO.pop(minPos)
            population.append(offspring.pop(minPos))

        return population

''' Main algorithm '''
class Genetic_Algorithm:

    def __init__(self, problem, initializer, selector, recombiner, mutator, replacer):
        self.times = problem.get_time()
        self.num_jobs = len(self.times)
        self.num_machines = problem.get_machine_num()
        self.population = initializer.initialize(self.num_jobs,self.num_machines)
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
            eval_machines = np.zeros(len(self.population))
            # calculate total processing time for every machine
            for index, job in enumerate(chromosome):
                eval_machines[job-1] = eval_machines[job-1] + self.times[index]

            # take max machine-time
            evaluation.append(max(eval_machines))

        self.evaluation = evaluation

    def evalutate_offspring(self, offspring):
        """ Evalutator """
        new_offspring = []
        # iterates all chromosomes in population and evaluates them
        for chromosome in offspring:
            # create a list of the machines
            eval_machines = np.zeros(len(self.population))
            # calculate total processing time for every machine
            for index, job in enumerate(chromosome):
                eval_machines[job - 1] = eval_machines[job - 1] + self.times[index]

            # take max machine-time
            new_offspring.append(max(eval_machines))
        return new_offspring

    def run_episode(self):
        """
        runs a whole episode of the GA
        """
        self.evaluate_population()
        selection = self.selector.select(self.population, self.evaluation)
        selection.extend(self.recombiner.recombine(selection))
        mutation = self.mutator.mutate(selection, self.num_machines)
        self.population = self.replacer.replace(self.population, mutation, self.evaluation, self.evalutate_offspring(mutation))


''' Hyperparameters '''
population_size = 100
selection_size = 20


''' Creation of Algorithm '''

problem_1 = Problem_1()
problem_2 = Problem_2()
problem_3 = Problem_3()

equal_initializer = Equal_Initializer(population_size)
random_initializer = Random_Initializer(population_size)

one_point_crossover = One_Point_Crossover()
two_point_crossover = Two_Point_Crossover()

lazy_mutator = Lazy_Mutator()
random_mutator = Random_Mutator()

roulette_wheel_selector = Roulette_Wheel_Selector(selection_size)
tournament_selector = Tournament_Selector(selection_size)

delete_all_replacer = Delete_All()
steady_replacer = Steady_State()

ga_1 = Genetic_Algorithm(problem_1, random_initializer, roulette_wheel_selector, one_point_crossover, random_mutator, steady_replacer)


''' Deployment '''

print('Start Deployment!')
mean_plot =[]

plt.plot(mean_plot)
for j in range(1000):
    # plotting each 1000 iterations
    for i in range(100):
        print('Step:', j+ i)
        ga_1.run_episode()
        ga_1.evaluate_population()
        # printing population
        print(ga_1.evaluation)
        print('fitness:', np.min(ga_1.evaluation))
        mean_plot.append(np.min(ga_1.evaluation))
    plt.plot(mean_plot)
    plt.show()
plt.show()