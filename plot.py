import numpy as np
import matplotlib.pyplot as plt

filename = 'PN:100 f:0.7 cr:0.5EXPONENTIAL'

best = np.loadtxt(filename + '_best')
mean = np.loadtxt(filename + '_mean')

axes = plt.gca()
axes.set_ylim([-1500000, 1500000])

plt.title(filename)

plt.xlabel('Iteration')
plt.ylabel('Profit')

plt.plot(mean, label='Mean')
plt.plot(best, label='Best')

plt.legend()

plt.show()