#!/usr/bin python

import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
import numpy as np
import math

def brownian_motion(D):
    dt = 1./D                              # compute time step
    B_inc = np.random.normal(0., 1., D)    # generate D random samples from N(0,1)
    B_inc = B_inc * np.sqrt(dt)            # compute each increment B_n = z_n*sqrt(dt)
    B = np.cumsum(B_inc)                   # sum them cumulatively in order to generate BM
    B = np.insert(B, 0, 0)                 # insert 0 element at the beginning of vector
    return B

# def poisson_process(D, l):    
#     dt = 1./D                              # compute time step
#     exp = np.random.exponential(1./l, D)   # generate D random samples from Exp(1/l)
#     exp = np.cumsum(exp)                   # sum them cumulatively
#     exp = np.insert(exp[:-1], 0, 0)        # insert 0 element at the beginning of vector
#                                            # and suppress last element
#     N = np.zeros(D)                        # start with N = [0, ..., 0]
#     # [0,1] here might be wrong though
#     LS = np.linspace(0, 1, num=D)          # create an evenly spaced vector in [0, 1]
#                                            # with D elements

#     # print("Lambda = " + str(l))
#     # print("exp = " + str(exp))
#     # print("LS = " + str(LS))

#     last_supremum = 0
#     for i in range(1, D): # i is for N[i]
#         satisfy_condition = []
#         for j in range(0, D): # j is for exp[j]
#             if exp[j] <= LS[i]*dt:
#                 satisfy_condition.append(j)
                
#         N[i] = max(satisfy_condition)
#         last_supremum = N[i]

#     # print(N)
#     return N

##                        
## 1. generate partitions
##
# e: epsilon
# R: left and right limits, [-R, e] and [e, R]
# D: total number of points considered, k = D/2
def LS_CGMY(e, R, D):
    LS1 = np.linspace(-R, -e, D/2)         # create an evenly spaced vector in [-R, e]
                                           # with D/2 elements
    LS2 = np.linspace(e, R, D/2)           # create an evenly spaced vector in [e, R]
                                           # with D/2 elements
    return (LS1, LS2)                      # return concatenation of the two vectors

##                        
## 2. compute \lambda_i's
##
# computes the Lévy measure of the CGMY process when x < 0
# interval corresponds to the tuple [a_{i-1}, a_i)
def v_CGMY_neg(C, G, M, Y, interval):
    result, _ = integrate.quad(lambda x: C*np.exp(G*x)*(-x**(-1-Y)), interval[0], interval[1])
    return result

# computes the Lévy measure of the CGMY process when x > 0
# interval corresponds to the tuple [a_i, a_{i+1})
def v_CGMY_pos(C, G, M, Y, interval):
    result, _ = integrate.quad(lambda x: C*np.exp(-M*x)*(x**(-1-Y)), interval[0], interval[1])
    return result

def compute_lambdas(C, G, M, Y, LS):
    neg_partitions = LS[0]                 # partitions where x < 0
    pos_partitions = LS[1]                 # partitions where x > 0

    neg_lambdas = []
    pos_lambdas = []
    
    for i in range(0, len(neg_partitions)-1):
        neg_lambdas.append(v_CGMY_neg(C, G, M, Y, (neg_partitions[i], neg_partitions[i+1])))
        
    for i in range(0, len(pos_partitions)-1):
        pos_lambdas.append(v_CGMY_pos(C, G, M, Y, (pos_partitions[i], pos_partitions[i+1])))

    return (np.array(neg_lambdas), np.array(pos_lambdas))

##
## 3. compute c_i's
##
def compute_c_neg(C, G, M, Y, interval):
    result, _ = integrate.quad(lambda x: (x**2)*C*np.exp(G*x)*(-x**(-1-Y)), interval[0], interval[1])
    return result

def compute_c_pos(C, G, M, Y, interval):
    result, _ = integrate.quad(lambda x: (x**2)*C*np.exp(-M*x)*(x**(-1-Y)), interval[0], interval[1])
    return result

def compute_cs(C, G, M, Y, LS, neg_lambdas, pos_lambdas):
    neg_partitions = LS[0]               # partitions where x < 0
    pos_partitions = LS[1]               # partitions where x > 0

    neg_cs = []
    pos_cs = []

    for i in range(0, len(neg_partitions)-1):
        neg_cs.append(math.sqrt((1./neg_lambdas[i])*compute_c_neg(C, G, M, Y, (neg_partitions[i], neg_partitions[i+1]))))

    for i in range(0, len(pos_partitions)-1):
        pos_cs.append(math.sqrt((1./pos_lambdas[i])*compute_c_pos(C, G, M, Y, (pos_partitions[i], pos_partitions[i+1]))))

    return (np.array(neg_cs), np.array(pos_cs))

##                        
## 4. compute \sigma^2(\epsilon)
##
def compute_sigma_squared_f(C, G, M, Y, epsilon):
    neg_result, _ = integrate.quad(lambda x: (x**2)*C*np.exp(G*x)*(-x**(-1-Y)), -epsilon, 0)
    pos_result, _ = integrate.quad(lambda x: (x**2)*C*np.exp(-M*x)*(x**(-1-Y)), 0, epsilon) 
    return neg_result + pos_result

##                        
## 5. simulate trajectories
##    
def CGMY(C, G, M, Y, gamma, sigma_squared, epsilon, R, D):
    
    K = int((D/2)-1)

    LS = LS_CGMY(epsilon, R, D)
    neg_lambdas, pos_lambdas = compute_lambdas(C, G, M, Y, LS)
    neg_cs, pos_cs = compute_cs(C, G, M, Y, LS, neg_lambdas, pos_lambdas)

    print("LS = ")
    print(LS)
    print("Lambda_is = ")
    print(neg_lambdas)
    print(pos_lambdas)
    print("C_is = ")
    print(neg_cs)
    print(pos_cs)

    # jumps smaller than epsilon
    B = brownian_motion(D)

    # jumps bigger than epsilon
    # generate D independent Poisson processes
    neg_ps = []
    pos_ps = []

    # reminder: each p is a np.array
    for i in range(0, K):
        # p = poisson_process(D, neg_lambdas[i])
        p = np.cumsum(np.random.poisson(neg_lambdas[i], K))
        neg_ps.append(p)

    for i in range(0, K):
        # p = poisson_process(D, pos_lambdas[i])
        p = np.cumsum(np.random.poisson(pos_lambdas[i], K))
        pos_ps.append(p)

    sigma_til = math.sqrt(sigma_squared + compute_sigma_squared_f(C, G, M, Y, epsilon))

    # finally computing trajectories
    X = np.zeros(D)
    small_jumps = np.zeros(D)
    big_jumps = np.zeros(D)

    # here we are assuming t = i (this might be wrong though)

    # for x < 0
    for i in range(0, K):
        for j in range(0, K):
            big_jumps[i] = neg_cs[j]*neg_ps[i][j]
            # if abs(neg_cs[j]) < 1:
            #     big_jumps[i] -= neg_lambdas[j]*i

    # for x > 0
    for i in range(0, K):
        for j in range(0, K):
            big_jumps[i+K] = pos_cs[j]*pos_ps[i][j]
            # if abs(pos_cs[j]) < 1:
            #     big_jumps[i+K] -= pos_lambdas[j]*i

    # trajectories when x < 0
    for i in range(0, K):
        X[i] = gamma*i + sigma_til*B[i] + big_jumps[i]
        small_jumps[i] = sigma_til*B[i]

    # trajectories when x > 0
    for i in range(0, K):
        X[i+K] = gamma*(i+K) + sigma_til*B[i+K] + big_jumps[i+K]
        small_jumps[i+K] = sigma_til*B[i+K]        

    return (X, small_jumps, big_jumps)

def plot(X, small_jumps, big_jumps):

    t = np.arange(len(X))

    df = pd.DataFrame({'X': X, 't': t, 'sj': small_jumps, 'bj': big_jumps})

    ax = plt.gca()
    
    df.plot(kind='line', x='t', y='sj', label='small jumps', ax=ax)
    df.plot(kind='line', x='t', y='bj', label='big jumps', ax=ax)
    df.plot(kind='line', x='t', y='X', label='X_t', ax=ax)
    
    plt.title('Simulation of X_t')
    plt.xlabel('t')
    plt.ylabel('Process')
    plt.show()

def main():
    np.random.seed(1)  # set seed

    C = 7
    G = 20
    M = 7
    Y = 0.01

    # C = 21.34
    # G = 49.78
    # M = 48.40
    # Y = 0.0037

    D = 100
    R = 0.01
    epsilon = 0.001
    sigma_squared = 0.5**2
    gamma = 0.001

    X, small_jumps, big_jumps = CGMY(C, G, M, Y, gamma, sigma_squared, epsilon, R, D)

    plot(X, small_jumps, big_jumps)

if __name__ == "__main__":
    main()

