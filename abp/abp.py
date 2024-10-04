import numpy as np
import argparse
import os

def main():

    #Get optional arguments
    parser = argparse.ArgumentParser(description='Compute an active Brownian particle trajectory')
    parser.add_argument('--v0', default=1.0, help='Self-propulsion velocity')
    parser.add_argument('--Dr', default=1.0, help='Rotational diffusion constant')
    parser.add_argument('--seed', default=123, help='Random seed')
    parser.add_argument('--nsteps', default=100000, help='Number of timesteps')
    parser.add_argument('--dt', default=1e-3, help='Timestep')
    parser.add_argument('--output_dir', default='data', help='Folder to output trajectories')
    parser.add_argument('--do_random_initial_velocity', default=0, help='1 for random initial velocity, otherwise points in z direction')
    args = parser.parse_args()

    #Set parameters
    v0 = float(args.v0)
    Dr = float(args.Dr)
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

    #Create arrays for instantaneous values of position (r), self-propulsion (p),
    #and angular velocity (omega)
    r = np.zeros(3)
    p = np.array([0.0,0.0,v0])
    omega = np.array([1.0,0.0,0.0])
    if do_random_initial_velocity==1:
        p = rng.standard_normal(size=3)
        p = v0*p/np.linalg.norm(p)

    #Propagate dynamics
    # dp/dt = omega x p (cross product)
    # where omega is a random angular velocity vector
    for n in range(nsteps):
        pos[n,:] = r
        self_prop_vel[n,:] = p
        r += dt*p #update position
        #Generate random angular velocity
        omega =  np.sqrt(2*Dr/dt)*rng.standard_normal(size=3)
        dp = dt*np.cross(omega, p)
        p += dp
        #print('dot:', np.dot(p,dp))
        #print('omega:', omega)
        #print('self-prop vel:', p)
        p = v0*p/np.linalg.norm(p) #need to re-normalize p due to finite time step
        
        disp[n,:] = r-pos[n,:]
        #print(np.linalg.norm(p))
        #print(p)

    #Save trajectory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_dir + '/traj_seed=%d.txt' % seed, np.c_[np.linspace(0,(nsteps-1)*dt, nsteps), pos, disp, self_prop_vel], 
               header = 'v0=%f, Dr=%f, dt=%f\nColumns: time, x, y, z, dx, dy, dz, px, py, pz' % (v0, Dr, dt))

if __name__=='__main__':
    main()