import numpy as np
import matplotlib.pyplot as plt
import itertools

def swap(entry):
    neighborhood = []

    mod = len(entry)

    for i in range(len(entry)):
        neighbor = entry.copy()

        last = i +1
        if i == mod - 1:
            last = 0

        neighbor[i] = entry[last]
        neighbor[last] = entry[i]

        neighborhood.append(neighbor)

    return neighborhood


def transposition(entry):
    neighbors = list(itertools.permutations(entry))
    neighbors.pop(0)
    return neighbors


class Knapsack:
    def __init__(self, num_items, max_weight=400):
        self.max_weight = max_weight
        self.values = self.get_values(num_items)
        self.select = [0] * num_items
        self.weights = self.get_weights(num_items)

    def get_values(self, num):
        return np.random.randint(1, high=1000, size=num)

    def get_weights(self, num):
        return np.random.randint(1, high=self.max_weight, size=num)

    def randomize_select(self):
        self.select = np.random.randint(0, 2, size=len(self.select))


def get_total_value(vals, selections):
    value = []
    for i in range(len(vals)):
        if selections[i] == 1:
            value.append(vals[i])
    return np.sum(value)


def check_weight_compatability(weights, selections, maxi):
    weight = []
    for i in range(len(weights)):
        if selections[i] == 1:
            weight.append(weights[i])
    total_weight = np.sum(weight)
    return 1 if total_weight < maxi else 0


def first_choice_hill_climb(sack, neighbor_type):
    weight_pass = 0
    while weight_pass == 0:
        weight_pass = check_weight_compatability(sack.weights, sack.select, sack.max_weight)
        if weight_pass == 0:
            print("Initial selection is invalid: Max weight exceeded!  Reseeding...")
            sack.randomize_select()
    print("SUCCESS")
    first_val = get_total_value(sack.values, sack.select)
    if neighbor_type == 'swap':
        new_vals = []
        new = swap(sack.select)
        for n in new:
            new_vals.append(get_total_value(sack.values, n))
        if max(new_vals) < first_val:
            return sack.select
        else:
            first_val = max(new_vals)
            sack.select = new_vals[np.argmax(new_vals)]


sack = Knapsack(5)
print(sack.values)
print(sack.weights)
sack.randomize_select()
print(sack.select)

first_choice_hill_climb(sack, 'swap')
