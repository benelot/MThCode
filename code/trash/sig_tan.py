import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.01)
y = (np.tanh(x)+1)/2
y2 = 1/(1 + np.exp(-x*2))
plt.plot(x, y2, label='sigmoid')
plt.plot(x, y, label='tanh')
plt.grid()
plt.legend()
plt.show()
