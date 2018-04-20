
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
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
    def random(self):
        return np.random.randint(1, self.num_machines + 1, self.num_jobs)
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


# In[9]:


i = Equal_Initializer(300, 20)
print(i.initialize())
print(len(i.initialize()))


# script
population_size = 100
problem_nr = 1

class Genetic_Algotihm:

    def __init__(self, initializer, selector, recombiner, mutator, replacer):
        self.inializer = initializer
        self.selector = selector
        self.recombiner = recombiner
        self.mutator = mutator
        self.replacer = replacer

    def generate_population(self, population_size, problem_nr, initializer):
        population = []
        for i in range(population_size):
            population.append(self.inializer.initialize())
        return population

ga = Genetic_Algotihm(i, i, i , i ,i)
print(ga.generate_population(population_size, 5, 3))

