import numpy as np
import matplotlib.pyplot as plt

def main():
    
    myfile = 'data/traj_seed=1.txt'
    data = np.loadtxt(myfile, skiprows=2)
    vel = data[::10,7:10]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(vel[:,0],vel[:,1],vel[:,2],linewidth=0.01)
    plt.savefig('self_prop_vel_traj.png')
    plt.show()
    
main()