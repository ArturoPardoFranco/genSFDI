import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.cos(2*np.pi*5*x + 0.2)

fig1 = plt.figure(1)
ax11 = fig1.add_subplot(1,1,1)
ax11.clear()
ax11.plot(x,y, 'g', linewidth=2, alpha=0.8)
ax11.grid(True)

plt.savefig('./test_output.png')
