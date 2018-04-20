import numpy as np
import random
import itertools
import time
import matplotlib.pyplot as plt

'''Knapsack Problem Hill Climbing
small neighbourhood: swap and transposition using intial random binary values
large neighbourhood: look at all combinations within constraint'''

MAX_WEIGHT = 400

w0 = [10,   300,  1,    200,  100]
v0 = [1000, 4000, 5000, 5000, 2000]


def random_values(num):
    return np.random.randint(1, high=1000, size=num)

def random_weights(num):
    return np.random.randint(1, high=MAX_WEIGHT//2, size=num)

def random_start(w):
    '''assign boolean random to start state, with constraint max_weight sum of w'''
    W = 0 #total weight
    full = False
    x = [0] * len(w) #decision which items to take
    while not full:
        for i in range(len(x)):
            r = random.randint(1,100) #gen random num
            if r <= 50:
                x[i] = 0 #don't take item
            elif r > 50:
                x[i] = 1 #take item
                W += w[i] #add weight to total
                if W >= MAX_WEIGHT: #if over limit
                    full = True
                    x[i] = 0 #remove that item
                    W -= w[i] #deduct the weight
    return x

def calc_value(array):
    ''' calculates total value of sack
    @param v: array of values
    @param array: binary choice value array'''
    V = 0
    for index, val in enumerate(array):
        if val == 1:
            V += v0[index]

    return V

def calc_weight(x):
    ''' calculates total weight of sack
    @param w: weight array
    @param x: binary choice value array'''
    W = 0
    for index, val in enumerate(x):
        if val == 1:
            W += w0[index]

    return W


def fchc_swap_neighbour(x):
    '''swap with neighbour to right if diff values, and if improves overall value (V)'''
    old_v = calc_value(x) #calculate current value of sack
    for i in range(len(x)):
         if x[i%len(x)] != x[(i+1) % len(x)]:
            r = x.copy() #copy sack
            r[i%len(x)], r[(i+1) % len(x)] = r[(i+1) % len(x)], r[i%len(x)] #switch the 2 diff neighbouring values
            v = calc_value(r)
            w = calc_weight(r)
            if  v > old_v and w <= MAX_WEIGHT: #if value is better than old one, and weight in limit
                x = r #set x as tmp bag r
                break
    return x

def hc_swap_neighbour(x):
    '''swap neighbour and pick best
        @param x: our array'''
    old_v = calc_value(x) #calculate current value of sack

    options = [] #store all viable options
    values = [] #zip later w their values

    for i in range(len(x)):
         if x[i%len(x)] != x[(i+1) % len(x)]:
            r = x.copy() #copy sack
            r[i%len(x)],r[(i+1) % len(x)] = r[(i+1) % len(x)], r[i%len(x)] #switch the 2 diff neighbouring values
            v = calc_value(r)
            w = calc_weight(r)
            if  v > old_v and w <= MAX_WEIGHT: #if value is better than old one, and weight in limit
                options.append(r) #add potential choices to our options
                values.append(v) #add the value of that option
    if values: # if it found something that is better, then...
        i = np.argmax(values) #find the max value
        x = options[i]      #set corresponding array to x and return

    return x



def fchc_transpose_neighbour(x):
    old_v = calc_value(x) #calculate current value of sack
    n = list(itertools.combinations(range(len(x)), 2)) #list of all possible 2 place swaps by index

    for tup in n: #for ea tuple of possible swap
        if x[tup[0]] != x[tup[1]]: #if the swap is not same value
            r = x.copy()
            r[tup[0]], r[tup[1]] = r[tup[1]], r[tup[0]] #make swap
            v = calc_value(r)
            w = calc_weight(r)
            if v > old_v and w <= MAX_WEIGHT:
                x = r
                break
    return x

def hc_transpose_neighbour(x):
    old_v = calc_value(x) #calculate current value of sack
    n = list(itertools.combinations(range(len(x)), 2)) #list of all possible 2 place swaps by index
    options = [] #store all viable options
    values = [] #store their values

    for tup in n: #for ea tuple of possible swap
        if x[tup[0]] != x[tup[1]]: #if the swap is not same value
            r = x.copy()
            r[tup[0]], r[tup[1]] = r[tup[1]], r[tup[0]] #make swap
            v = calc_value(r)
            w = calc_weight(r)
            if v > old_v and w <= MAX_WEIGHT:
                options.append(r) #add potential choices to our options
                values.append(v) #add the value of that option
    if values:
        i = np.argmax(values) #find the max value
        x = options[i]      #set corresponding array to x and return

    return x

if __name__ == '__main__':
    example = False
    if example:
        bag = random_start(w0)
        w1 = calc_weight(bag)
        v1 = calc_value(bag)
    else:
        MAX_WEIGHT = 50000 # 10000
        num_items = 40 # 20
        w0 = random_weights(num_items)
        v0 = random_values(num_items)
        bag = random_start(w0)
        w1 = calc_weight(bag)
        v1 = calc_value(bag)

    print('\nInitial selection:\t', bag)
    print('Initial weights:\t', w0)
    print('Initial values:\t', v0)
    print('Initial value ' + str(v1))
    print('Initial weight ' + str(w1))

    iterations_fchfs = []
    values_fchfs = []
    times_fchfs = []

    iterations_hfs = []
    values_hfs = []
    times_hfs = []

    iterations_fchft = []
    values_fchft = []
    times_fchft = []

    iterations_hft = []
    values_hft = []
    times_hft = []

    print("\nOrder: fchc_swap, hc_swap, fchc_tranpose, hc_transpose")
    repeat = 10
    for _ in range(repeat):
        output = bag.copy()
        start = time.time()
        for i in range(100):
            new_output = fchc_swap_neighbour(output)
            if new_output == output:
                iterations_fchfs.append(i)
                times_fchfs.append(time.time()-start)
                values_fchfs.append(calc_value(new_output))
                break
            else:
                output = new_output

        output = bag.copy()
        start = time.time()
        for i in range(100):
            new_output = hc_swap_neighbour(output)
            if new_output == output:
                iterations_hfs.append(i)
                times_hfs.append(time.time()-start)
                values_hfs.append(calc_value(new_output))
                break
            else:
                output = new_output

        output = bag.copy()
        start = time.time()
        for i in range(100):
            new_output = fchc_transpose_neighbour(output)
            if new_output == output:
                iterations_fchft.append(i)
                times_fchft.append(time.time()-start)
                values_fchft.append(calc_value(new_output))
                break
            else:
                output = new_output


        output = bag.copy()
        start = time.time()
        for i in range(100):
            new_output = hc_transpose_neighbour(output)
            if new_output == output:
                iterations_hft.append(i)
                times_hft.append(time.time()-start)
                values_hft.append(calc_value(new_output))
                break
            else:
                output = new_output

    print("FCHC-S")
    print("iterations\n", np.mean(iterations_fchfs))
    print("final_times\n", np.mean(times_fchfs))
    print("final_values\n", np.mean(values_fchfs))
    print()

    print("HC-S")
    print("iterations\n", np.mean(iterations_hfs))
    print("final_times\n", np.mean(times_hfs))
    print("final_values\n", np.mean(values_hfs))
    print()

    print("FCHC-T")
    print("iterations\n", np.mean(iterations_fchft))
    print("final_times\n", np.mean(times_fchft))
    print("final_values\n", np.mean(values_fchft))
    print()

    print("HC-T")
    print("iterations\n", np.mean(iterations_hft))
    print("final_times\n", np.mean(times_hft))
    print("final_values\n", np.mean(values_hft))
    print()