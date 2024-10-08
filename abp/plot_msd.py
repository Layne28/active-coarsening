import numpy as np
import matplotlib.pyplot as plt

def abp_theory(t, Dr, v0):
    return v0**2/(2*Dr**2)*(2*Dr*t + np.exp(-2*Dr*t)-1)

v_abp_theory = np.vectorize(abp_theory)
data = np.loadtxt('msd.txt')

fig = plt.figure()
plt.plot(data[:,0], data[:,1], label='simulation')
plt.plot(data[:,0], v_abp_theory(data[:,0], 1.0, 1.0), linewidth=0.9, color='black', linestyle='--', label='theory')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle$')
plt.savefig('msd.png')