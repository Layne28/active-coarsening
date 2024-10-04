import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('msd.txt')

fig = plt.figure()
plt.plot(data[:,0], data[:,1], label='simulation')
plt.plot(data[:,0], 10*data[:,0], linewidth=0.9, color='black', linestyle='--', label=r'$t$')
plt.plot(data[:,0], 10*data[:,0]**2, linewidth=0.9, color='red', linestyle='--', label=r'$t^2$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle$')
plt.savefig('msd.png')