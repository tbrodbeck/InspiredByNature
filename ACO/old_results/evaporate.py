
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import copy

from random import choices


# In[14]:


class AntColonyTSPOptimizer:
    def __init__(self, ants, evaporation, intensification, alpha=1, beta=0, beta_evaporation_rate = 0, choose_best=.1):
        self.ants = ants
        self.evaporation_rate = evaporation
        self.pheromone_intensification = intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.choose_best = choose_best
        self.mean = []
        self.best = []
        self.beta_evaporation_rate = beta_evaporation_rate

    def _get_coordinate_matrix(self, num_cities):
        '''
        Stores all possibile pathes between all cities
        '''

        coords = []
        for i in range(num_cities):
            for j in range(num_cities):
                coords.append((i, j))
        return coords

    def _get_destination_matrix(self, num_cities):
        '''
        Stores all destinations for each city as an array
        '''

        destinations = []
        for i in range(num_cities):
            row = [i + 1 for i in range(num_cities)]
            destinations.append(row)
        return np.asarray(destinations)

    def _get_eta_matrix(self, tsp_map):
        '''
        Generates the eta-heuristic matrix (the diagonal is not needed)
        '''

        return self._remove_diagonal((1 / tsp_map) ** self.heuristic_beta)

    def _get_pheromone_matrix(self, num_cities):
        '''
        Fills the pheromone matrix initially with ones (the diagonal is not needed)
        '''

        pheromone_matrix = self._remove_diagonal(np.ones((num_cities, num_cities)))
        return pheromone_matrix ** self.heuristic_alpha

    def _initialize(self, num_cities, tsp_map):

        self.pheromone_matrix = self._get_pheromone_matrix(num_cities)
        self.coordinate_matrix = self._get_coordinate_matrix(num_cities)
        self.destination_matrix = self._get_destination_matrix(num_cities)
        self.eta_matrix = self._get_eta_matrix(tsp_map)

    def _remove_diagonal(self, matrix):
        '''
        Replaces the values of the diagonal of a matrix by zeros
        '''

        remove_diagonal = np.eye(len(matrix))
        matrix[remove_diagonal == 1] = 0
        return matrix

    def _get_probabilities(self, from_city, run, divide=True):
        '''
        Calculates the probabilites to go from a certain city to each other possible city
        '''

        probability = []
        for to_city in range(len(run)):
            top = run[from_city, to_city] * self.eta_matrix[from_city, to_city]
        # calculates the probabilty via the normal formula
            if divide:
                bottom = np.sum(run[from_city] * self.eta_matrix[from_city])
            # calculation of the probabilities uses only the numerator of the formula (in case q0 is used)
            else:
                bottom = 1
            probability.append(top / bottom)
        return probability

    def _delete_city(self, run, city):
        '''
        Deletes every city that has already been visited from the current path matrix
        '''

        for i in range(len(self.destination_matrix)):
            for j in range(len(self.destination_matrix)):
                if self.destination_matrix[i, j] == city + 1:
                    run[i, j] = 0
        return run

    def _stack_probabilities(self, probs):
        '''
        Changes the format of the probabilities from a sequence of 1-D array to a single 2-D array
        '''

        probability = np.column_stack(([p for p in probs]))
        return probability


    def _explore(self, tsp_map):
        """
        This function generates a number of ants (self.ants) and makes them run through the graph
        ants are only permitted to visit each city only once
        """
        # list containing the path of all ants of an iteration
        routes = []
        # same as above but contains pairs [(1,2),(2,3),...]
        coordinate_routes = []
    
        for ant in range(self.ants):
            # current_run contains the pheromone values of unvisited cities
            current_run = copy.deepcopy(self.pheromone_matrix)
            # lists containing information about the route of a certain ant
            route = []
            coordinates = []
            # determine starting point + delete it from current run
            original_city = np.random.randint(0, len(self.pheromone_matrix))
            current_run = self._delete_city(current_run, original_city)
            # save the starting point
            route.append(original_city)
            current_city = copy.deepcopy(original_city)

            for i in range(len(self.pheromone_matrix) - 1):  # -1 because initial city already chosen
                # if our random number is smaller than the choose best probability
                # then the function _get_probabilities chooses just the best option for pheromones (divisor = 1)
                if np.random.random() < self.choose_best:
                    probability = self._get_probabilities(current_city, current_run, divide=False)
                    next_city = np.argmax(probability)
                # else the function computes the divident in order to take the distances into account
                else:
                    probability = self._get_probabilities(current_city, current_run, divide=True)
                    next_city = np.random.choice(range(len(probability)), p=probability)

                # save the next city
                route.append(next_city)
                index = self._get_index(current_city, next_city, len(current_run))
                coordinates.append(self.coordinate_matrix[index])
                # remove the next city from our "to-do list" and update the current city pointer
                current_run = self._delete_city(current_run, next_city)
                current_city = next_city

            # get from the last city back to the origin
            route.append(original_city)
            index = self._get_index(current_city, original_city, len(current_run))
            coordinates.append(self.coordinate_matrix[index])

            # save the results for each ant into the arrays for the whole iteration & return
            routes.append(route)
            coordinate_routes.append(coordinates)
        return routes, coordinate_routes

    # function to get the index of a certain touple in a given ist (list is computed in _get_coordinateMatrix
    def _get_index(self, i, j, length):
        down = length * i
        return down + j

    # get the total distance for every route of a certain iteration
    def _evaluate_solutions(self, routes, tsp_map):
        scores = []
        # sum the distances for every route
        for route in routes:
            score = 0
            for city in route:
                score += tsp_map[city]
            scores.append(score)
        return scores
    
    def _evaporate(self):
        '''
        Reduces the pheromone values by the evaporation rate
        '''
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        self.heuristic_beta *= (1 - self.beta_evaporation_rate)
        
    def _intensify(self, best):
        '''
        Increases the pheromone values accroding to the best solution by the pheromone intensification value
        '''
        for city in best:
            self.pheromone_matrix[city] += self.pheromone_intensification
        self._remove_diagonal(self.pheromone_matrix)
            
    def fit(self, tsp_map, iterations=None, verbose=True):
        """
        This functions calls the above defined functions in the correct order
        """
        # Initialization
        tsp_map = self._remove_diagonal(tsp_map)
        num_cities = len(tsp_map)
        self._initialize(num_cities, tsp_map)

        # do the iterations
        if iterations:
            for iteration in range(iterations):
                routes, coords = self._explore(tsp_map)
                scores = self._evaluate_solutions(coords, tsp_map)
                self._evaporate()

                # save the best distance and the average of all distances of an iteration
                self.best.append(np.min(scores))
                self.mean.append(np.mean(scores))

                self._intensify(coords[np.argmin(scores)])
                if verbose:
                    print("SCORES\n", scores)
                    print("BEST ROUTE\n", coords[np.argmin(scores)])
                    print("Iteration:\t", iteration)


module_name = '01.tsp'

module = np.loadtxt(module_name)

op = AntColonyTSPOptimizer(ants=10, evaporation=.1, intensification=0.1, alpha=1, beta=1, beta_evaporation_rate=.05,choose_best=.1)
op.fit(tsp_map=module, iterations=200)

meanlist = op.mean
bestlist = op.best

configuration_name = 'evap=10' + module_name
np.savetxt('mean_' + configuration_name, meanlist)
np.savetxt('best_' + configuration_name, bestlist)

