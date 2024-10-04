import numpy as np
import argparse
import os
import numba

def main():

    #Get optional arguments
    parser = argparse.ArgumentParser(description='Compute mean-squared displacement of AOUP trajectories')
    parser.add_argument('--input_dir', default='data', help='Directory to look for trajectories')
    parser.add_argument('--tmax', default=100.0, help='Max time difference for computing MSD')
    parser.add_argument('--freq', default=10, help='Sampling frequency')
    args = parser.parse_args()

    #Set parameters
    input_dir = args.input_dir
    tmax = float(args.tmax)
    freq = int(args.freq)
    msd_all = np.zeros(1)

    #Loop through files to compute MSD
    files = [f for f in os.listdir(input_dir) if 'traj_seed=' in f]
    for f in files:
        print(f)
        with open(input_dir + '/' + f) as myfile:
            line = myfile.readline()
            dt = float(line.split('dt=')[-1])
        data = np.loadtxt(input_dir + '/' + f, skiprows=2)
        pos = data[::freq,1:4]
        nmax = int(tmax/(freq*dt))
        msd = np.zeros(nmax)
        msd = compute_msd(msd, pos, nmax)
        print(msd)
        if msd_all.size==1:
            msd_all = msd
        else:
            print('adding')
            msd_all += msd
    msd_all /= len(files)

    np.savetxt('msd.txt', np.c_[np.linspace(0,tmax-dt,nmax), msd_all])

@numba.jit(nopython=True)
def compute_msd(msd, pos, nmax):
    for t in range(nmax):
        for mu in range(3):
            msd[t] += np.mean((pos[:-nmax,mu] - pos[t:(-nmax+t),mu])**2)
    return msd

if __name__=='__main__':
    main()