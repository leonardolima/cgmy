#!/usr/bin python

import numpy as np
import math

def brownianMotion(N):
    dt = 1./N                          # compute time step
    Binc = np.random.normal(0., 1., N) # generate N random samples from N(0,1)
    Binc = Binc * np.sqrt(dt)          # compute each increment B_n = z_n*sqrt(dt)
    B = np.cumsum(Binc)                # sum them cumulatively in order to generate BM
    B = np.insert(B, 0, 0)             # insert 0 element at the beginning of vector
    return B

def poissonProcess(N, l):
    dt = 1./N                            # compute time step
    exp = np.random.exponential(1./l, N) # generate N random samples from Exp(1/l)
    exp = np.cumsum(exp)                 # sum them cumulatively
    exp = np.insert(exp, 0, 0)           # insert 0 element at the beginning of vector
    P = np.zeros(N)                      # start with N = [0, ..., 0]

    lastSupIndex = 0                     # keep the index of last supremum
    for i in range(1, N):                # iterate over P on [1, ..., N]
        for j in range(lastSupIndex, N): # iterate over exp on [lastSupIndex, ..., N]
            if exp[j] > i*dt:            # check if already reached supremum
                if j != 0:               # if yes and j != 0, the supremum is j-1
                    P[i] = j-1
                    lastSupIndex = j-1
                    break

                if j == 0:               # special case when j == 0
                    P[i] = 0

    return P

def main():
    np.random.seed(2019)  # set seed
    N = 10                # number of increments
    B = brownianMotion(N)
    P = poissonProcess(N, 10)

if __name__ == "__main__":
    main()
    
