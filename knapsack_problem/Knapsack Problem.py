
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import itertools


# In[ ]:


def swap(entry):
    neighbors = []
    neighbor = []
    mod = len(entry)
    
    for i in range(len(entry)):
        last = i + 1
        if i == mod - 1:
            last = 0
        neighbor = entry.copy()
        neighbor[last], neighbor[i] = entry[i], entry[last]
        neighbors.append(neighbor)
    return neighbors


# In[441]:


def transposition(entry):
    neighbors = list(set(itertools.permutations(entry)))
    neighbors.pop(-1)
    return neighbors


# In[442]:


transposition([1,0,0,0,1])


# In[ ]:

def check_weight_compatability(weights, selections, maxi):
    weight = []
    for i in range(len(weights)):
        print('Selections:', selections)
        if selections[i] == 1:
            weight.append(weights[i])
    total_weight = np.sum(weight)
    return 1 if total_weight < maxi else 0

class Knapsack:
    '''
    we first want to define the problem
    '''
    def __init__(self, num_items, given=0, max_weight=400):
        self.max_weight = max_weight
        self.values = self.get_values(num_items)
        self.weights = self.get_weights(num_items)
        self.select = self.randomize_select(given)
        
    def get_values(self, num):
        return np.random.randint(1, high=1000, size=num)
    
    def get_weights(self, num):
        return np.random.randint(1, high=self.max_weight, size=num)
    
    def randomize_select(self, given=0):
        if given != 0:
            if check_weight_compatability(self.weights, given, self.max_weight):
                self.select = given
            else:
                print('error: given select does not fit maxweight')

        else:
            self.select = np.random.randint(0, 2, size=len(self.values))
            if not check_weight_compatability(self.weights, self.select, self.max_weight):
                self.randomize_select()


# In[458]:


def get_total_value(vals, selections):
    value = []
    for i in range(len(vals)):
        if selections[i] == 1:
            value.append(vals[i])
    return np.sum(value)


def hill_climb2(sack, neighbor_type, hillclimb):
    finished = False
    iteration = 0
    first_val = get_total_value(sack.values, sack.select)
    while not finished:
        if neighbor_type == 'swap':
            neighbor = swap(sack.select)
        if neighbor_type == 'transpose':
            neighborhood = transposition(sack.select)
        for i in reversed(range(len(neighbor_type))):
            if not check_weight_compatability(sack.weights, neighborhood[i], sack.max_weight):
                neighborhood.pop(i)
        if len(neighborhood) == 0:
            print("ALL OVERWEIGHT")
            return False
        new_vals = []
        for n in neighborhood:
            new_val = get_total_value(sack.values, n)
            if new_val > first_val:
                if hillclimb == 'first_choice':
                    sack.select = n
                    return hill_climb2(sack, neighbor_type, hillclimb)
                if hillclimb == 'general':
                    new_vals.append(new_val)
        if hillclimb == 'general':
            max_val = max(new_vals)
            for index, v in enumerate(new_vals):
                if v == max_val:
                    sack.select = neighborhood[index]
                    return hill_climb2(sack, neighbor_type, hillclimb)
        finished = True



def hill_climb(sack, neighbor_type):
    if neighbor_type == 'swap':
        found = False
        iterations = 0
        first_val = get_total_value(sack.values, sack.select)
        while not found:
#             print("Swapping neighborhood...")
            new_vals = []
            new = swap(sack.select)
            for i in reversed(range(len(new))):
                if not check_weight_compatability(sack.weights, new[i], sack.max_weight):
#                     print("Overweight!!")
                    new.pop(i)
            if len(new) == 0:
                print("ALL OVERWEIGHT")
                return False, False
            for n in new:
                new_vals.append(get_total_value(sack.values, n))
            maximum = max(new_vals)
#             print("Maximum new value: {} (Old value: {})".format(maximum, first_val))
            if maximum <= first_val:
#                 print("Selecting first neighborhood!")
                return sack.select, iterations
            else:
                first_val = maximum
#                 print("Selecting second neighborhood for further swapping!")
                sack.select = new[np.argmax(new_vals)]
                iterations += 1
    if neighbor_type == 'transpose':
        found = False
        iterations = 0
        first_val = get_total_value(sack.values, sack.select)
        while not found:
#             print("Transposing neighborhood...")
            new_vals = []
            new = transposition(sack.select)
            for i in reversed(range(len(new))):
                if not check_weight_compatability(sack.weights, new[i], sack.max_weight):
#                     print("Overweight!!")
                    new.pop(i)
            if len(new) == 0:
                print("ALL OVERWEIGHT")
                return False, False
            for n in new:
                new_vals.append(get_total_value(sack.values, n))
            maximum = max(new_vals)
#             print("Maximum new value: {} (Old value: {})".format(maximum, first_val))
            if maximum <= first_val:
#                 print("Selecting first neighborhood!")
                return sack.select, iterations
            else:
                first_val = maximum
#                 print("Selecting second neighborhood for further transposing!")
                sack.select = new[np.argmax(new_vals)]
                iterations += 1

def first_choice_hill_climb(sack, neighbor_type):
    weight_pass = 0
    while weight_pass == 0:
        weight_pass = check_weight_compatability(sack.weights, sack.select, sack.max_weight)
        if weight_pass == 0:
#             print("Initial selection is invalid: Max weight exceeded!  Reseeding...")
            sack.randomize_select()
    if neighbor_type == 'swap':  
        found = False
        iterations = 0
        first_val = get_total_value(sack.values, sack.select)
        while not found:
#             print("Swapping neighborhood...")
            new_vals = []
            new = swap(sack.select)
            for i in reversed(range(len(new))):
                if not check_weight_compatability(sack.weights, new[i], sack.max_weight):
#                     print("Overweight!!")
                    new.pop(i)
            if len(new) == 0:
                print("ALL OVERWEIGHT")
                return False, False
            for i, n in enumerate(new):
                if get_total_value(sack.values, n) > first_val:
                    first_val = get_total_value(sack.values, n)
#                     print("Selecting {}th neighborhood for further swapping!".format(i))
                    sack.select = n
                    iterations += 1
                    break
                else:
#                     print("Selecting first neighborhood!")
                    return sack.select, iterations
    if neighbor_type == 'transpose':
        found = False
        iterations = 0
        first_val = get_total_value(sack.values, sack.select)
        while not found:
#             print("Transposing neighborhood...")
            new_vals = []
            new = transposition(sack.select)
            for i in reversed(range(len(new))):
                if not check_weight_compatability(sack.weights, new[i], sack.max_weight):
#                     print("Overweight!!")
                    new.pop(i)
            if len(new) == 0:
                print("ALL OVERWEIGHT")
                return False, False
            for i, n in enumerate(new):
                if get_total_value(sack.values, n) > first_val:
                    first_val = get_total_value(sack.values, n)
#                     print("Selecting {}th neighborhood for further transposing!".format(i))
                    sack.select = n
                    iterations += 1
                    break
                else:
#                     print("Selecting first neighborhood!")
                    return sack.select, iterations


# In[ ]:


sack = Knapsack(5)
print(sack.values)
print(sack.weights)
print(sack.select)
print(sack)
hill_climb2(sack, 'swap', 'first_choice')


# In[ ]:


sack = Knapsack(5)
print(sack.values)
print(sack.weights)
sack.randomize_select()
print(sack.select)
hill_climb(sack, 'transpose')


# In[ ]:


sack = Knapsack(5)
print(sack.values)
print(sack.weights)
sack.randomize_select()
print(sack.select)
first_choice_hill_climb(sack, 'swap')
print("==SWITCH==")
sack = Knapsack(5)
print(sack.values)
print(sack.weights)
sack.randomize_select()
print(sack.select)
first_choice_hill_climb(sack, 'transpose')


# In[ ]:


import time


# In[465]:


def trial_run(runs=100):
    sack = Knapsack(5, [1,0,0,0,1], max_weight=777)
    vals = []
    times = []
    iterations = []
    for i in range(runs):
        times.append(time.time())
        slct, itr = hill_climb(sack, 'swap')
        if isinstance(slct, bool):
            times[i] = time.time() - times[i]
            continue
        val = get_total_value(sack.values, slct)
        vals.append(val)
        iterations.append(itr)
        print("THE SELECTION", sack.select)
        print("THE VALUES", sack.values)
        print("THE WEIGHTS", sack.weights)
        sack.select = [1,0,0,0,1]
        times[i] = time.time() - times[i]
    print(vals, times, iterations)


# In[467]:


trial_run()


# In[446]:


test = []
if len(test) == 0:
    print('yay')

