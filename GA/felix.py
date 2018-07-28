import numpy
import time

class Chromosome():

    def __init__(self, array):
        self.Allele = array
        
    def printself(self):  
        
        print(self.Allele)  
        
        
class Fitness():
    
    def __init__(self,scenario):
        
        if scenario == 0:
            
            self.m = 20
            self.values = numpy.zeros(300)
            
            for i in range(0,200):
                self.values[i] = numpy.random.randint(10,1001)
            for i in range(200,300):
                self.values[i] = numpy.random.randint(100,301)
                
        elif scenario == 1:
            
            self.m=20
            self.values = numpy.zeros(300)
            
            for i in range(0,150):
                self.values[i] = numpy.random.randint(10,1001)
            for i in range(150,300):
                self.values[i] = numpy.random.randint(400,701)
                
        else:
            
            self.m=50
            self.values = numpy.zeros(101)
            self.values[0] = 50
            
            for i in range(0,50):
                self.values[1+2*i] = i+50
                self.values[2+2*i] = i+50
            
            
    def detFitness(self,Chromosome):
    
        if len(Chromosome.Allele) != len(self.values):
            return False
    
        machineLoad = numpy.zeros(self.m)
    
        for i in range(0,len(Chromosome.Allele)):
            
            machineLoad[int(Chromosome.Allele[i]-1)] += self.values[i]
        
        return max(machineLoad)


class Initializer():
    
    def __init__(self, scenario, pop_size):
        
        self.Chromosomes = set()
        self.scenario = scenario
        self.pop_size = pop_size
         
    def initialize(self):
        
        self.Chromosomes = set()
        
        if self.scenario == 0 or self.scenario == 1:
            
            for p in range(0,self.pop_size):
                c = numpy.zeros(300)
                
                for i in range(0,len(c)):
                    c[i]= numpy.random.randint(0,21)
                
                chromo = Chromosome(c)
                self.Chromosomes.add(chromo)
                
            return self.Chromosomes
            
        else:
            
            for p in range(0,self.pop_size):
            
                c = numpy.zeros(101)
                
                for i in range(0,len(c)):
                    
                    c[i]= numpy.random.randint(0,51)
                    
                chromo = Chromosome(c)
                self.Chromosomes.add(chromo)
                
            return self.Chromosomes
            
            
class Selector():
    
    def __init__(self, matingNumber):
        
        self.matingPool = set()
        self.matingNumber = matingNumber
        
    def select(self, population, fit):
        
        self.matingPool = set()
        
        stats = numpy.zeros((2,len(population)))
        
        j = 0
        total = 0
        
        list = []
        for i in population:
            
            list.append(i)
            stats[0,j] = fit.detFitness(i)
            total += stats[0,j]
            j+=1
        
        cumulate = 0
        
        for i in range(0,len(population)):
            
            cumulate += stats[0,i]
            stats[1,i] = cumulate/total
            
        for i in range(0,self.matingNumber):
            
            r = numpy.random.ranf()
            j = 0
            
            while r > stats[1,j] and j < len(population):
                j+=1
                
            self.matingPool.add(list[j])
            
        return self.matingPool
             
            
class Recombiner():
    
    def __init__(self, crossoverProb):
        
        self.offspring = set()
        self.crossoverProb = crossoverProb
        
    def onePointCrossover(self, matingPool):
        
        self.offspring = set()
        
        matingList = []
        
        for i in matingPool:
            matingList.append(i)
            
        while len(matingList)>0:
            offspring1 = matingList.pop(numpy.random.randint(0,len(matingList))).Allele
            offspring2 = matingList.pop(numpy.random.randint(0,len(matingList))).Allele
        
            r = numpy.random.ranf()
        
            if r <= self.crossoverProb:
            
                pos = numpy.random.randint(0,len(offspring1))
            
                for i in range(pos,len(offspring1)):
                
                    mem = offspring1[i]
                    offspring1[i] = offspring2[i]
                    offspring2[i] = mem
            
            chromosomeOne = Chromosome(offspring1)
            chromosomeTwo = Chromosome(offspring2)
        
            self.offspring.add(chromosomeOne)
            self.offspring.add(chromosomeTwo)
        
        return self.offspring
        
        
            
class Mutator():
    
    def __init__(self, mutationProb):
        
        self.Chromosomes = set()
        self.mutationProb = mutationProb
        
    #def mutate(self):
        
class Replacer():
    
    def __init__(self):
        
        self.newPop = set()
        
    def deleteALL(self, population, offspring):
        
        if len(offspring) < len(population):
            return False
        
        self.newPop = set()
        
        offspringList = []
        
        for i in offspring:
            
            list.append(i)
        
        for i in range(0,len(population)):  
            offspringList.pop(numpy.random.randint(0,len(offspringList)))
            
        
        
        
#def AssignmentSearch(scenario, pop_size, mutation_prob, time_limit, verbose):
def AssignmentSearch(scenario, pop_size, mutationProb, crossoverProb, time_limit):
            
    matingNumber = round(pop_size/10)
    matingNumber *= 2
    
    init = Initializer(scenario, pop_size)
    fit = Fitness(scenario)
    select = Selector(matingNumber)
    recombiner = Recombiner(crossoverProb)
    mutator = Mutator(mutationProb)
    
    population = init.initialize()
    
    
    
    currentTime = time.time()
    
    #while time.time()-currentTime < time_limit:
    matingPool = select.select(population, fit, matingNumber)
    offspring = recombiner.onePointCrossover(matingPool)
    
    
    #for i in matingPool:
    #    print(i.Allele)

AssignmentSearch(0,10,0.5,0.5,10)