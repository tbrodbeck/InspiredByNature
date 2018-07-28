import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# fixed values
LARGE = 1000000000

def demand(price, maxPrice, maxDemand):
    """
    @author: Nico Potyka
    """
    # if price is greater than max price, return 0
    if price > maxPrice:
        return 0
    # if product is free return maxDemand (ignore negative price)
    if price <= 0:
        return maxDemand
    # else determine demand based on price
    demand = maxDemand - price**2 * maxDemand / maxPrice**2
    return demand

def cost(x, kwhPerPlant, costPerPlant, maxPlants):
    """
    @autor: Nico Potyka
    """
    # if x is non-positive, return 0
    if x <= 0:
        return 0
    # if x is greater than what can be generated return prohibitively large value
    if x > kwhPerPlant * maxPlants:
        return LARGE
    # otherwise determine number of plants needed to generate x
    plantsNeeded = math.ceil(x /kwhPerPlant)
    # cost is number of plants needed times cost per plant
    return plantsNeeded * costPerPlant


# for each of 3 markets: maximum price at which customers buy, maximum demand
pm1 = 0.45
dm1 = 2000000
pm2 = 0.25
dm2 = 30000000
pm3 = 0.2
dm3 = 20000000

# for each of 3 plants: kWh per plant, cost per plant, maximum number of plants that can be used
kp1 = 50000
cp1 = 10000
mp1 = 100
kp2 = 600000
cp2 = 80000
mp2 = 50
kp3 = 4000000
cp3 = 400000
mp3 = 3

def objective(chromosome):
    """
    :param chromosome: chromosome of our DE
    :return: profit
    """
    # parameters must not be negative
    for val in chromosome:
        if val < 0:
            return -LARGE
    # readout values
    e1 = chromosome[0]
    e2 = chromosome[1]
    e3 = chromosome[2]
    s1 = chromosome[3]
    s2 = chromosome[4]
    s3 = chromosome[5]
    p1 = chromosome[6]
    p2 = chromosome[7]
    p3 = chromosome[8]
    # formulas from the slides
    revenue = min(demand(p1, pm1, dm1), s1) * p1 + min(demand(p2, pm2, dm2),s2) * p2 + min(demand(p3, pm3, dm3), s3) * p3
    production_cost = cost(e1, kp1, cp1, mp1) + cost(e2, kp2, cp2, mp2) + cost(e3, kp3, cp3, mp3)
    purchasing_cost = max(s1 + s2 + s3 - e1 - e2 - e3, 0) * 0.6
    costs = production_cost + purchasing_cost
    return revenue - costs

def initialize(population_size):
    """
    initialize a population of chromosomes
    :param population_size: number of chromosomes
    :return: initialized population
    """
    population = []
    for i in range (population_size):
        chromosome = []
        # we take for the energy the maximum demand in corresponding market
        chromosome.append(random.random() * dm1)
        chromosome.append(random.random() * dm2)
        chromosome.append(random.random() * dm3)
        # for the selling value we take the maximum demand in corresponding market
        chromosome.append(random.random() * dm1)
        chromosome.append(random.random() * dm2)
        chromosome.append(random.random() * dm3)
        # and for the price the maximum price of corresponding market
        chromosome.append(random.random() * pm1)
        chromosome.append(random.random() * pm2)
        chromosome.append(random.random() * pm3)
        population.append(chromosome)
    return np.asanyarray(population)

def mutate(population, f):
    """
    :param population:
    :param f: usually between 0.4 and 1
    :return: donor-vector
    """
    population_size = len(population)
    chromosomes = np.random.choice(population_size, (3))
    indices = np.random.permutation(3)
    # creates donor according to equation (2) in Differential Evolution (2011)
    return population[chromosomes[indices[0]]] + f * ( population[chromosomes[indices[1]]] - population[chromosomes[indices[2]]] )

def cross(chromosome, mutation, cr, method):
    """
    crossover of a chromosome- and a mutation-vector of same dimension
    :param population:
    :param mutation:
    :param cr: 0 <= cr <= 1
    :param method: 'EXPONENTIAL' | 'BINOMIAL'
    :return: trial-vector
    """
    trial = np.empty(np.shape(chromosome))

    if method == 'EXPONENTIAL':
        n = np.random.choice(population_size)
        # generating l according to chapter II,C in Differential Evolution (2011)
        l = 0
        while ( random.random() <= cr ) & ( l <= population_size ):
            l += 1
        # crossover according to equation (4) in DE(2011)
        for j, value in enumerate(chromosome):
            if j == n%population_size:
                trial[j] = mutation[j]
            else:
                trial[j] = chromosome[j]
        return trial

    elif method == 'BINOMIAL':
        # crossover according to equation (5) in DE(2011)
        j_rand = np.random.choice(population_size)
        for j, value in enumerate(chromosome):
            if (random.random() <= cr) | j == j_rand:
              trial[j] = mutation[j]
            else:
                trial[j] = chromosome[j]
        return trial

    else:
        print('only input \'EXPONENTIAL\' || \'BINOMIAL\'')

if __name__ == '__main__':
    """
    according to Algorithm 1 in DE(2011)
    """
    # hyperparameters
    population_size = 100
    f = 0.4
    #f_decay_rate = 1 - 0.001
    cr = 0.2
    crossover_method = 'BINOMIAL'
    # initialization
    done = False
    step = 0
    # store old mean value for stop-criterion
    old_mean = -999999999
    population = initialize(population_size)
    plot_mean = []
    plot_max = []
    # main loop
    while not done:
        values = []
        for i in range(population_size):
            target = population[i]
            donor = mutate(population, f)
            trial = cross(target, donor, cr, crossover_method)
            # selection according to equation (6) in DE(2011)
            value_trial = objective(trial)
            value_target = objective(target)
            if value_trial >= value_target:
                population[i] = trial
                values.append(value_trial)
            else:
                population[i] = target
                values.append(value_target)
        mean = np.mean(values)
        maximum = max(values)
        plot_mean.append(mean)
        plot_max.append(maximum)
        step += 1
        print('Step:', step, 'Best:', maximum, 'Mean:', mean)
        # stop-criterion: if mean does not change after x steps
        if step%100 == 0:
            if mean > old_mean:
                old_mean = mean
            else:
                done = True
        # if f > 0.4:
        #     f = f * f_decay_rate
    # before shutting down
    # print some values
    for chromosome in population:
        print(objective(chromosome))
    print(population)
    filename = 'PN' + str(population_size) + '_f' + str(f) + '_cr' + str(cr) + '_' + crossover_method
    print(filename)
    # save results
    np.savetxt(filename + '_mean', plot_mean)
    np.savetxt(filename + '_best', plot_max)
    # and show plot
    axes = plt.gca()
    axes.set_ylim([0, 1500000])
    plt.title(filename)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Solution')
    plt.plot(plot_mean, label='Mean')
    plt.plot(plot_max, label='Best')
    plt.legend()
    plt.show()
