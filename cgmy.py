#!/usr/bin python

import scipy.integrate as integrate
import numpy as np
import math

def brownian_motion(N):
    dt = 1./N                              # compute time step
    B_inc = np.random.normal(0., 1., N)    # generate N random samples from N(0,1)
    B_inc = B_inc * np.sqrt(dt)            # compute each increment B_n = z_n*sqrt(dt)
    B = np.cumsum(B_inc)                   # sum them cumulatively in order to generate BM
    B = np.insert(B, 0, 0)                 # insert 0 element at the beginning of vector
    return B

def poisson_process(N, l):
    dt = 1./N                              # compute time step
    exp = np.random.exponential(1./l, N)   # generate N random samples from Exp(1/l)
    exp = np.cumsum(exp)                   # sum them cumulatively
    exp = np.insert(exp[:-1], 0, 0)        # insert 0 element at the beginning of vector
                                           # and suppress last element
    P = np.zeros(N)                        # start with P = [0, ..., 0]
    LS = np.linspace(0, 1, num=N)          # create an evenly spaced vector in [0, 1]
                                           # with N elements

    last_sup_index = 0                     # keep the index of last supremum
    for i in range(1, N):                  # iterate over P on [1, ..., N]
        for j in range(last_sup_index, N): # iterate over exp on [lastSupIndex, ..., N]
            if exp[j] > LS[i]*dt:          # check if already reached supremum
                if j != 0:                 # if yes and j != 0, the supremum is j-1
                    P[i] = j-1
                    last_sup_index = j-1
                    break

                if j == 0:                 # special case when j == 0
                    P[i] = 0

    return P

# e: epsilon
# R: left and right limits, [-R, e] and [e, R]
# D: total number of points considered, k = D/2
def LS_CGMY(e, R, D):
    LS1 = np.linspace(-R, -e, D/2)       # create an evenly spaced vector in [-R, e]
                                         # with D/2 elements
    LS2 = np.linspace(e, R, D/2)         # create an evenly spaced vector in [e, R]
                                         # with D/2 elements
    return (LS1, LS2)                    # return concatenation of the two vectors

# computes the Lévy measure of the CGMY process when x < 0
# interval corresponds to the tuple [a_{i-1}, a_i)
def v_CGMY_neg(C, G, M, Y, interval):
    result, err = integrate.quad(lambda x:  C*np.exp(G*x)*(-x**(-1-Y)), interval[0], interval[1])
    return result

# computes the Lévy measure of the CGMY process when x > 0
# interval corresponds to the tuple [a_i, a_{i+1})
def v_CGMY_pos(C, G, M, Y, interval):
    result, err = integrate.quad(lambda x:  C*np.exp(-M*x)*(x**(-1-Y)), interval[0], interval[1])
    return result

def compute_lambdas(C, G, M, Y, LS):
    neg_partitions = LS[0]               # partitions where x < 0
    pos_partitions = LS[1]               # partitions where x > 0

    neg_lambdas = []
    pos_lambdas = []
    
    for i in range(0, len(neg_partitions)-1):
        neg_lambdas.append(v_CGMY_neg(C, G, M, Y, (neg_partitions[i], neg_partitions[i+1])))
        
    for i in range(0, len(pos_partitions)-1):
        pos_lambdas.append(v_CGMY_pos(C, G, M, Y, (pos_partitions[i], pos_partitions[i+1])))

    return (np.array(neg_lambdas), np.array(pos_lambdas))

# CGMY Lévy density 
# def k_CGMY(C, G, M, Y, LS):
#     ls1 = LS[LS < 0]                     # select all elements from LS s.t. < 0
#     ls2 = LS[LS == 0]                    # select all elements from LS s.t. == 0
#     ls3 = LS[LS > 0]                     # select all elements from LS > 0

#     # Implementation of equation (7) of the paper
#     for i in range(0, len(ls1)):
#         ls1[i] = C*(math.exp(-G*abs(ls1[i]))/(abs(ls1[i])**(1+Y))) # LS < 0

#     for i in range(0, len(ls3)):
#         ls3[i] = C*(math.exp(-M*abs(ls3[i]))/(abs(ls3[i])**(1+Y))) # LS > 0

#     return [ls1, ls2, ls3]
    
# def CGMY(C, G, M, Y, LS):
    # jumps smaller than epsilon
    # B = brownianMotion(N)
    # sigma = 2*(ep**2)*
    # jumps larger than epsilon

def main():
    np.random.seed(2019)  # set seed

    D = 200
    R = 1
    epsilon = 0.001
    C = 5
    G = 7
    M = 23
    Y = 0.5

    LS = LS_CGMY(epsilon, R, D)
    lambdas = compute_lambdas(C, G, M, Y, LS)
    print(lambdas)
    
    

if __name__ == "__main__":
    main()
    
