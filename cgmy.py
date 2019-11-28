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
    exp = np.insert(exp[:-1], 0, 0)      # insert 0 element at the beginning of vector
                                         # and suppress last element
    P = np.zeros(N)                      # start with P = [0, ..., 0]
    LS = np.linspace(0, 1, num=N)        # create an evenly spaced vector in [0, 1]
                                         # with N elements

    lastSupIndex = 0                     # keep the index of last supremum
    for i in range(1, N):                # iterate over P on [1, ..., N]
        for j in range(lastSupIndex, N): # iterate over exp on [lastSupIndex, ..., N]
            if exp[j] > LS[i]*dt:        # check if already reached supremum
                if j != 0:               # if yes and j != 0, the supremum is j-1
                    P[i] = j-1
                    lastSupIndex = j-1
                    break

                if j == 0:               # special case when j == 0
                    P[i] = 0

    return P

def LSCGMY(C, G, M, Y, e, M, N):
    LS1 = linspace(-M, -e, N/2)          # create an evenly spaced vector in [-M, e]
                                         # with N/2 elements
    LS2 = linspace(e, M, N/2)            # create an evenly spaced vector in [e, M]
                                         # with N/2 elements
    return LS1 + LS2                     # return concatenation of the two vectors

# CGMY Lévy density 
def k_CGMY(C, G, M, Y, LS):
    ls1 = LS[LS < 0]                     # select all elements from LS < 0
    ls2 = LS[LS == 0]                    # select all elements from LS == 0
    ls3 = LS[LS > 0]                     # select all elements from LS > 0

    # Implementation of equation (7) of the paper
    for i in range(0, len(ls1)):
        ls1[i] = C*(math.exp(-G*abs(ls1[i]))/(abs(ls1[i])**(1+Y))) # LS < 0

    for i in range(0, len(ls3)):
        ls3[i] = C*(math.exp(-M*abs(ls3[i]))/(abs(ls3[i])**(1+Y))) # LS > 0

    return ls1 + ls2 + ls3
    
# def CGMY(C, G, M, Y, LS):
    # jumps smaller than epsilon
    # B = brownianMotion(N)
    # sigma = 2*(ep**2)*
    # jumps larger than epsilon

def main():
    np.random.seed(2019)  # set seed
    N = 10                # number of increments, must be divisible by 2
    B = brownianMotion(N)
    P = poissonProcess(N, 10)
    

if __name__ == "__main__":
    main()
    
