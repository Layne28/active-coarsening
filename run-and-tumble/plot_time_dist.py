import numpy as np
import matplotlib.pyplot as plt

def main():
    
    myfile = 'data/times_seed=1.txt'
    data = np.loadtxt(myfile)
    hist, bins = np.histogram(data, density=True)
    bins = (bins[:-1]+bins[1:])/2
    
    fig = plt.figure()
    plt.plot(bins, hist, color='blue', label='simulation')
    plt.plot(bins, np.exp(-bins), color='red', label='theory')
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$P(\tau)$')
    plt.legend()
    plt.savefig('switch_time_histogram.png')
    plt.show()
    
main()