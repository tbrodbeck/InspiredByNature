import numpy as np
import matplotlib.pyplot as plt
import copy

from random import choices


class AntColonyTSPOptimizer:
    def __init__(self, ants, evaporation, intensification, alpha=1, beta=0, pheromone_evaporation_rate = 0, choose_best=.1):
        self.ants = ants
        self.evaporation_rate = evaporation
        self.pheromone_intensification = intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.choose_best = choose_best
        self.mean = []
        self.best = []
        self.pheromone_evaporation_rate = pheromone_evaporation_rate
        
    def _get_coordinate_matrix(self, num_cities):
        coords = []
        for i in range(num_cities):
            for j in range(num_cities):
                coords.append((i, j))
        return coords
        
    def _get_destination_matrix(self, num_cities):
        destinations = []
        for i in range(num_cities):
            row = [i + 1 for i in range(num_cities)]
            destinations.append(row)
        return np.asarray(destinations)
    
    def _get_eta_matrix(self, tsp_map):
        return self._remove_diagonal((1 / tsp_map) ** self.heuristic_beta)
    
    def _get_pheromone_matrix(self, num_cities):
        pheromone_matrix = self._remove_diagonal(np.ones((num_cities, num_cities)))
        return pheromone_matrix ** self.heuristic_alpha
    
    def _initialize(self, num_cities, tsp_map):
        self.pheromone_matrix = self._get_pheromone_matrix(num_cities)
        self.coordinate_matrix = self._get_coordinate_matrix(num_cities)
        self.destination_matrix = self._get_destination_matrix(num_cities)
        self.eta_matrix = self._get_eta_matrix(tsp_map)

    def _remove_diagonal(self, matrix):
        remove_diagonal = np.eye(len(matrix))
        matrix[remove_diagonal==1] = 0
        return matrix
    
    def _get_probabilities(self, from_city, run, divide=True):
        probability = []
        for to_city in range(len(run)):
            top = run[from_city, to_city] * self.eta_matrix[from_city, to_city]
            if divide:
                bottom = np.sum(run[from_city] * self.eta_matrix[from_city])
            else:
                bottom = 1
            probability.append(top / bottom)
        return probability

    def _delete_city(self, run, city):
        for i in range(len(self.destination_matrix)):
            for j in range(len(self.destination_matrix)):
                if self.destination_matrix[i, j] == city + 1:
                    run[i, j] = 0
        return run
            
    def _stack_probabilities(self, probs):
        probability = np.column_stack(([p for p in probs]))
        return probability
        
    def _explore(self, tsp_map):
        routes = []
        coordinate_routes = []
    
        for ant in range(self.ants):
            current_run = copy.deepcopy(self.pheromone_matrix)
            route = []
            coordinates = []
            original_city = np.random.randint(0, len(self.pheromone_matrix))
            current_run = self._delete_city(current_run, original_city)

            route.append(original_city)
            current_city = copy.deepcopy(original_city)
            for i in range(len(self.pheromone_matrix) - 1):  # -1 because initial city already chosen
                if np.random.random() < self.choose_best:
                    probability = self._get_probabilities(current_city, current_run, divide=False)
                    next_city = np.argmax(probability)
                else:
                    probability = self._get_probabilities(current_city, current_run, divide=True)
                    next_city = np.random.choice(range(len(probability)), p=probability)
                route.append(next_city)
                index = self._get_index(current_city, next_city, len(current_run))
                coordinates.append(self.coordinate_matrix[index])
                current_run = self._delete_city(current_run, next_city)
                current_city = next_city

            route.append(original_city)
            index = self._get_index(current_city, original_city, len(current_run))
            coordinates.append(self.coordinate_matrix[index])

            routes.append(route)
            coordinate_routes.append(coordinates)
        return routes, coordinate_routes

    def _get_index(self, i, j, length):
        down = length * i
        return down + j
    
    def _evaluate_solutions(self, routes, tsp_map):
        scores = []
        for route in routes:
            score = 0
            for city in route:
                score += tsp_map[city]
            scores.append(score)
        return scores
    
    def _evaporate(self):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        self.heuristic_beta *= (1 - self.pheromone_evaporation_rate)
        
    def _intensify(self, best):
        for city in best:
            self.pheromone_matrix[city] += self.pheromone_intensification
        self._remove_diagonal(self.pheromone_matrix)

    def fit(self, tsp_map, iterations=None, verbose=True):
        tsp_map = self._remove_diagonal(tsp_map)
        num_cities = len(tsp_map)
        self._initialize(num_cities, tsp_map)
        if iterations:
            for iteration in range(iterations):
                routes, coords = self._explore(tsp_map)
                scores = self._evaluate_solutions(coords, tsp_map)
                self._evaporate()
                
                self.best.append(np.min(scores))
                self.mean.append(np.mean(scores))

                self._intensify(coords[np.argmin(scores)])
                if verbose:
                    print("SCORES\n", scores)
                    print("BEST ROUTE\n", coords[np.argmin(scores)])
                    print("Iteration:\t", iteration)

module_name = '01.tsp'
first = np.loadtxt(module_name)

op = AntColonyTSPOptimizer(ants=10, evaporation=.1, intensification=1, alpha=1, beta=1, pheromone_evaporation_rate=.1,choose_best=.1)
op.fit(first, 200)

meanlist = op.mean
bestlist = op.best

configuration_name = 'listEvap10' + module_name
np.savetxt('mean_' + configuration_name, meanlist)
np.savetxt('best_' + configuration_name, bestlist)
