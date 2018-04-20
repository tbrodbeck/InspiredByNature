import numpy as np
import random
import itertools
import time
import matplotlib.pyplot as plt

'''Knapsack Problem Hill Climbing
small neighbourhood: swap and transposition using intial random binary values
large neighbourhood: look at all combinations within constraint'''

w = [10,   300,  1,    200,  100]
v = [1000, 4000, 5000, 5000, 2000]
def random_start(w):
    '''assign boolean random to start state, with constraint max 400 sum of w'''
    W = 0 #total weight
    full = False
    x = [0, 0, 0, 0, 0] #decision which items to take
    while not full:
        for i in range(len(x)):
            r = random.randint(1,100) #gen random num
            if r <= 50:
                x[i] = 0 #don't take item
            elif r > 50:
                x[i] = 1 #take item
                W += w[i] #add weight to total
                if W >= 400: #if over limit
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
            V += v[index]

    return V

def calc_weight(x):
    ''' calculates total weight of sack
    @param w: weight array
    @param x: binary choice value array'''
    W = 0
    for index, val in enumerate(x):
        if val == 1:
            W += w[index]

    return W


def fchc_swap_neighbour(x):
    '''swap with neighbour to right if diff values, and if improves overall value (V)'''
    old_v = calc_value(x) #calculate current value of sack
    for i in range(len(x)):
         if x[i%5] != x[(i+1) % 5]:
            r = x.copy() #copy sack
            r[i%5], r[(i+1) % 5] = r[(i+1) % 5], r[i%5] #switch the 2 diff neighbouring values
            v = calc_value(r)
            w = calc_weight(r)
            if  v > old_v and w <= 400: #if value is better than old one, and weight in limit
                x = r #set x as tmp bag r
    return x

def hc_swap_neighbour(x):
    '''swap neighbour and pick best
        @param x: our array'''
    old_v = calc_value(x) #calculate current value of sack

    options = [] #store all viable options
    values = [] #zip later w their values

    for i in range(len(x)):
         if x[i%5] != x[(i+1) % 5]:
            r = x.copy() #copy sack
            r[i%5],r[(i+1) % 5] = r[(i+1) % 5], r[i%5] #switch the 2 diff neighbouring values
            v = calc_value(r)
            w = calc_weight(r)
            if  v > old_v and w <= 400: #if value is better than old one, and weight in limit
                options.append(r) #add potential choices to our options
                values.append(v) #add the value of that option
    if values: # if it found something that is better, then...
        i = np.argmax(values) #find the max value
        x = options[i]      #set corresponding array to x and return

    return x



def fchc_transpose_neighbour(x):
    old_v = calc_value(x) #calculate current value of sack
    n = random.sample(range(0, 4), 2) #generates 2 diff numbers 0 - 4

    for i in range(5): #try to get a better value this many times
        r = x.copy() #copy our array
        if r[n[0]] != r[n[1]]: #if the 2 random items are not same
            r[n[0]], r[n[1]] = r[n[1]], r[n[0]] #swap the 2 random spots
            v = calc_value(r)
            w = calc_weight(r)
            if v > old_v and w <= 400:
                x = r
                break #break out of loop once find 1st better value

    return x

def hc_transpose_neighbour(x):
    old_v = calc_value(x) #calculate current value of sack
    n = list(itertools.combinations(range(5), 2)) #list of all possible 2 place swaps by index
    options = [] #store all viable options
    values = [] #store their values

    for tup in n: #for ea tuple of possible swap
        if x[tup[0]] != x[tup[1]]: #if the swap is not same value
            r = x.copy()
            r[tup[0]], r[tup[1]] = r[tup[1]], r[tup[0]] #make swap
            v = calc_value(r)
            w = calc_weight(r)
            if v > old_v and w <= 400:
                # print("APPENDING")
                options.append(r) #add potential choices to our options
                values.append(v) #add the value of that option
    if values:
        i = np.argmax(values) #find the max value
        # print("neighbours: " + str(options)) #*
        x = options[i]      #set corresponding array to x and return

    return x

if __name__ == '__main__':
    bag = random_start(w)
    w1 = calc_weight(bag)
    v1 = calc_value(bag)

    print('value ' + str(v1))
    print('weight ' + str(w1))

    iterations = []
    final_values = []
    final_outs = []
    final_times = []
    final_weights = []

    print("Order: fchc_swap, hc_swap, fchc_tranpose, hc_transpose")
    output = bag.copy()
    start = time.time()
    for i in range(100):
        new_output = fchc_swap_neighbour(output)
        if new_output == output:
            print('v',calc_value(new_output), 'w',calc_weight(new_output))
            print(new_output)
            print()
            iterations.append(i)
            final_times.append(time.time()-start)
            final_values.append(calc_value(new_output))
            final_outs.append(new_output)
            final_weights.append(calc_weight(new_output))
            break
        else:
            output = new_output

    output = bag.copy()
    start = time.time()
    for i in range(100):
        new_output = hc_swap_neighbour(output)
        if new_output == output:
            print('v',calc_value(new_output), 'w',calc_weight(new_output))
            print(new_output)
            print()
            iterations.append(i)
            final_times.append(time.time()-start)
            final_values.append(calc_value(new_output))
            final_outs.append(new_output)
            final_weights.append(calc_weight(new_output))
            break
        else:
            output = new_output

    output = bag.copy()
    start = time.time()
    for i in range(100):
        new_output = fchc_transpose_neighbour(output)
        if new_output == output:
            print('v',calc_value(new_output), 'w',calc_weight(new_output))
            print(new_output)
            print()
            iterations.append(i)
            final_times.append(time.time()-start)
            final_values.append(calc_value(new_output))
            final_outs.append(new_output)
            final_weights.append(calc_weight(new_output))
            break
        else:
            output = new_output


    output = bag.copy()
    start = time.time()
    for i in range(100):
        new_output = hc_transpose_neighbour(output)
        if new_output == output:
            print('v',calc_value(new_output), 'w',calc_weight(new_output))
            print(new_output)
            print()
            iterations.append(i)
            final_times.append(time.time()-start)
            final_values.append(calc_value(new_output))
            final_outs.append(new_output)
            final_weights.append(calc_weight(new_output))
            break
        else:
            output = new_output

