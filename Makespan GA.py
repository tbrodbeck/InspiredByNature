
# coding: utf-8

# In[12]:


import numpy as np
from abc import ABC, abstractmethod


# In[2]:


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


# In[55]:

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


# In[ ]:


class Evaluator():
    def evaluate():
        pass


# In[56]:


class Selector():
    def roulette(self):
        raise NotImplementedError
    def tournament(self, candidates):
        return max(candidates)



class Genetic_Algotihm:

    def __init__(self, initializer, selector, recombiner, mutator, replacer):
        self.inializer = initializer
        self.selector = selector
        self.recombiner = recombiner
        self.mutator = mutator
        self.replacer = replacer

        self.population = initializer.initialize()

    def evaluate_population(self):
        pass

''' hyperparameters '''
population_size = 1
num_jobs = 300
num_machines = 20

''' main script '''
equal_initializer = Equal_Initializer(num_jobs,num_machines,population_size)
random_initializer = Random_Initializer(num_jobs,num_machines,population_size)

ga_equal = Genetic_Algotihm(equal_initializer, 0,0,0,0)
ga_random = Genetic_Algotihm(random_initializer,0,0,0,0)
print(ga_equal.population)
print(ga_random.population)

