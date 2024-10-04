import numpy as np
import argparse
import os

def main():

    #Get optional arguments
    parser = argparse.ArgumentParser(description='Compute an run-and-tumble particle trajectory')
    parser.add_argument('--v0', default=1.0, help='swim velocity')
    parser.add_argument('--tau', default=1.0, help='run time')
    parser.add_argument('--seed', default=123, help='Random seed')
    parser.add_argument('--nsteps', default=100000, help='Number of timesteps')
    parser.add_argument('--dt', default=1e-3, help='Timestep')
    parser.add_argument('--output_dir', default='data', help='Folder to output trajectories')
    parser.add_argument('--do_random_initial_velocity', default=0, help='1 for random initial velocity, otherwise points in z direction')
    args = parser.parse_args()

    #Set parameters
    v0 = float(args.v0)
    tau = float(args.tau)
    seed = int(args.seed)
    nsteps = int(args.nsteps)
    dt = float(args.dt)
    output_dir = args.output_dir
    do_random_initial_velocity = int(args.do_random_initial_velocity)
    
    #Create random number generator
    rng = np.random.default_rng(seed)

    #Make arrays for storing trajectory
    pos = np.zeros((nsteps, 3))
    self_prop_vel = np.zeros((nsteps, 3))
    disp = np.zeros((nsteps, 3)) #displacement at each timestep

    #Create arrays for instantaneous values of position (r) and self-propulsion (p),
    r = np.zeros(3)
    p = np.array([0.0,0.0,v0])
    times = [] #list of switching times
    time_curr = 0
    counter = 0
    if do_random_initial_velocity==1:
        p = rng.standard_normal(size=3)
        p = v0*p/np.linalg.norm(p)

    #Propagate dynamics
    # randomly change the swim direction
    # after time sampled exponential distribution
    for n in range(nsteps):
        pos[n,:] = r
        self_prop_vel[n,:] = p
        r += dt*p #update position
        time_curr += dt
        #Decide whether to reorient
        prob = np.exp(-dt/tau)
        if rng.uniform()>prob:
            #redraw velocity vector
            p = rng.standard_normal(size=3)
            p = v0*p/np.linalg.norm(p)
            counter += 1
            times.append(time_curr)
            time_curr = 0
        
        disp[n,:] = r-pos[n,:]
        #print(np.linalg.norm(p))
        #print(p)

    print('switched %d times' % counter)
    #Save trajectory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_dir + '/traj_seed=%d.txt' % seed, np.c_[np.linspace(0,(nsteps-1)*dt, nsteps), pos, disp, self_prop_vel], 
               header = 'v0=%f, tau=%f, dt=%f\nColumns: time, x, y, z, dx, dy, dz, px, py, pz' % (v0, tau, dt))
    np.savetxt(output_dir + '/times_seed=%d.txt' % seed, times)

if __name__=='__main__':
    main()