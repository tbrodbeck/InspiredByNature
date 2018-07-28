
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import copy

from random import choices


# In[14]:


class AntColonyTSPOptimizer:
    def __init__(self, ants, evaporation, intensification, alpha=1, beta=1, choose_best=.05):
        self.ants = ants
        self.evaporation_rate = evaporation
        self.pheromone_intensification = intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.choose_best = choose_best
        
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
#             print("TOP", top)
            if divide:
#                 print("*****\n",run[from_city],"\n",self.eta_matrix[from_city],
#                       "\n",run[from_city] * self.eta_matrix[from_city],"\n******")
                bottom = np.sum(run[from_city] * self.eta_matrix[from_city])
#                 print("BOTTOM", bottom)
            else:
                bottom = 1
            probability.append(top / bottom)
        return probability

    def _delete_city(self, run, city):
        #print('destm', run)
        for i in range(len(self.destination_matrix)):
            for j in range(len(self.destination_matrix)):
                if self.destination_matrix[i, j] == city + 1:  # cause coords are 0 index based
                    run[i, j] = 0
        return run
            
    def _stack_probabilities(self, probs):
        probability = np.column_stack(([p for p in probs]))
        return probability
        
    def _explore(self, tsp_map):
        routes = []
        coordinate_routes = []
    
        # DEBU0
#         self.choose_best = 0
    
        for ant in range(self.ants):
            current_run = copy.deepcopy(self.pheromone_matrix)
            route = []
            coordinates = []
            original_city = np.random.randint(0, len(self.pheromone_matrix))
            current_run = self._delete_city(current_run, original_city)
#             print("=========================\nINITIAL CITY", initial_city)
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

#             print(route)
#         print(routes)
        return routes, coordinate_routes

    def _get_index(self, i, j, length):
        down = length * j
        return down + i
    
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
        
    def _intensify(self, best):
        # print('ITENSIFY', best)
        for city in best:
            self.pheromone_matrix[city] += self.pheromone_intensification
        self._remove_diagonal(self.pheromone_matrix)
        # print('INFIED', self.pheromone_matrix)
            
    def fit(self, tsp_map, iterations=None, verbose=True):
        tsp_map = self._remove_diagonal(tsp_map)
        num_cities = len(tsp_map)
        self._initialize(num_cities, tsp_map)
        if iterations:
            for iteration in range(iterations):
                self.heuristic_beta = self.heuristic_beta * 0.99
                routes, coords = self._explore(tsp_map)
                #print(routes, coords)
                scores = self._evaluate_solutions(coords, tsp_map)
                self._evaporate()

#                 print(coords[np.argmax(scores)])
                self._intensify(coords[np.argmin(scores)])
               # print(self.pheromone_matrix)
                if verbose:
                    print("SCORES\n", scores)
                    print("BEST ROUTE\n", coords[np.argmin(scores)])
                    print("Iteration:\t", iteration)
#                     print("Minimum Distance:\t", min(scores))
#                     print(scores)
                    
                
        


# In[15]:


first = np.loadtxt('01.tsp')
test = np.random.randint(1,100, (3,3))
print(test)




# In[16]:

#
# example = [[0,9,9,5],[9,0,5,9],[5,9,0,9],[9,5,9,0]]
# example = np.asarray(example)
# print(example)
#
#
# # In[17]:
#
#
op = AntColonyTSPOptimizer(16, .05, 1)
op.fit(first, 1000)


# In[ ]:

'''
print(first[1,1])


# In[ ]:


one = np.ones((10,10))
one[0,1] = 2
two = np.ones((10,10)) * 2

print(one)

print(np.sum(one[0]))


# In[ ]:


probs = [[1,2,3],[4,5,6],[7,8,9]]

probability = np.column_stack(([p for p in probs]))
print(probability)


# In[ ]:


print(probability.shape)


# In[ ]:


def lol(num_cities):
        coordinates = []
        for i in range(num_cities):
            row = [1 + i for i in range(num_cities)]
            coordinates.append(row)
        return np.asarray(coordinates)


# In[ ]:


print(lol(10).shape)


# In[ ]:


print(1/test)


# In[ ]:


i = [1,2,3]
ii = [2,2,2]

print(np.array(i) * np.array(ii))

'''