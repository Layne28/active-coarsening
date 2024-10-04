import numpy as np
import matplotlib.pyplot as plt

def main():
    
    myfile = 'data/traj_seed=1.txt'
    data = np.loadtxt(myfile, skiprows=2)
    vel = data[:,7:10]
    pos = data[:,1:4]
    
    fig = plt.figure()
    quiver_freq=1000
    ax = fig.add_subplot(projection='3d')
    ax.plot(pos[:,0],pos[:,1],pos[:,2],linewidth=1.0)
    #ax.quiver(pos[::quiver_freq,0],pos[::quiver_freq,1],pos[::quiver_freq,2],vel[::quiver_freq,0],vel[::quiver_freq,1],vel[::quiver_freq,2],color='red', linewidth=0.5)
    #ax.plot(vel[:,0],vel[:,1],vel[:,2],linewidth=0.01)
    plt.show()
    
main()