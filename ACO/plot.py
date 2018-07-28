'''
Created on 12.05.2018

@author: Felix
'''
import numpy as np
import matplotlib.pyplot as plt

configuration_name = '_results3_01.tsp'
best = np.loadtxt('best' + configuration_name)
mean = np.loadtxt('mean' + configuration_name)

plt.xlabel('Iteration')
plt.ylabel('Solution')
plt.plot(mean)
plt.plot(best)

plt.show()