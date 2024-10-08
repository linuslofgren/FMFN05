import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1.4, 1000)
# g = np.pow(x, 3) -x+np.sqrt(2)/2

def g(x):
    return np.pow(x, 3) -x+np.sqrt(2)/2
# g_p = 3*np.pow(x, 2)-1

def g_p(x):
    return 3*np.pow(x, 2)-1

x_j = 0.7

x_js = []
gs = []
for _ in range(15):
    old_x_j = x_j
    old_g = g(x_j)
    x_j = x_j - (g(x_j)/g_p(x_j))
    x_js.append(x_j)
    gs.append(g(x_j))
    plt.arrow(old_x_j, old_g, x_j-old_x_j, g(x_j)-old_g, head_width=0.01)

# plt.plot(x_js, gs)
plt.plot(x, g(x))
plt.show()