#!/usr/bin python

from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
import numpy as np
import math

def brownian_motion(D):
    dt = 1./D
    delta = D/5
    B = norm.rvs(size=D, scale=delta*math.sqrt(dt))
    B = np.insert(B[:-1], 0, 0)
    B = np.cumsum(B)
    return B

##                        
## 1. generate partitions
##
# alpha: lower and upper limits
# D: total number of points considered, k = D/2
def LS_CGMY(alpha, D):
    K = int((D/2)+1)

    LS1 = np.zeros(K)
    LS2 = np.zeros(K)

    for i in range(1, K+1):
       LS1[i-1] =  -alpha/i
       LS2[K-i] = alpha/i

    return (LS1, LS2)

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
    neg_partitions = LS[0] # partitions where x < 0
    pos_partitions = LS[1] # partitions where x > 0

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
    neg_partitions = LS[0] # partitions where x < 0
    pos_partitions = LS[1] # partitions where x > 0

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
def CGMY(C, G, M, Y, gamma, sigma_squared, D, alpha):
    
    K = int(D/2)
    LS = LS_CGMY(alpha, D)
    neg_partitions = LS[0]
    pos_partitions = LS[1]
    neg_lambdas, pos_lambdas = compute_lambdas(C, G, M, Y, LS)
    neg_cs, pos_cs = compute_cs(C, G, M, Y, LS, neg_lambdas, pos_lambdas)

    # print("LS = " + str(LS))
    # print("neg_lambdas = " + str(neg_lambdas))
    # print("neg_cs = " + str(neg_cs))

    # jumps smaller than epsilon
    B = brownian_motion(D)

    # jumps bigger than epsilon
    # generate D independent Poisson processes
    neg_ps = []
    pos_ps = []

    # reminder: each p is a np.array
    for i in range(0, K):
        p = np.cumsum(np.random.poisson(neg_lambdas[i], K))
        neg_ps.append(p)

    # print("neg_ps = " + str(neg_ps))

    for i in range(0, K):
        p = np.cumsum(np.random.poisson(pos_lambdas[i], K))
        pos_ps.append(p)
    
    epsilon = pos_partitions[0]
    sigma_til = math.sqrt(sigma_squared + compute_sigma_squared_f(C, G, M, Y, epsilon))

    # finally computing trajectories
    neg_X = np.zeros(K)
    pos_X = np.zeros(K)

    neg_small_jumps = np.zeros(K)
    pos_small_jumps = np.zeros(K)

    neg_big_jumps = np.zeros(K)
    pos_big_jumps = np.zeros(K)

    # for x < 0
    for i in range(0, K):
        for j in range(0, K):
            neg_big_jumps[i] += neg_cs[j]*neg_ps[i][j]
            if abs(neg_cs[j]) < 1:
                neg_big_jumps[i] -= neg_lambdas[j]*neg_partitions[i]

    # print(neg_big_jumps)

    # for x > 0
    for i in range(0, K):
        for j in range(0, K):
            pos_big_jumps[i] += pos_cs[j]*pos_ps[i][j]
            if abs(pos_cs[j]) < 1:
                pos_big_jumps[i] -= pos_lambdas[j]*pos_partitions[i]

    # trajectories when x < 0
    for i in range(0, K):
        neg_X[i] = gamma*neg_partitions[i] + sigma_til*B[i] + neg_big_jumps[i]
        neg_small_jumps[i] = sigma_til*B[i]

    # trajectories when x > 0
    for i in range(0, K):
        pos_X[i] = gamma*pos_partitions[i] + sigma_til*B[i+K] + pos_big_jumps[i]
        pos_small_jumps[i] = sigma_til*B[i+K]

    # concatenating vectors
    X = np.concatenate([neg_X, pos_X])
    small_jumps = np.concatenate([neg_small_jumps, pos_small_jumps])
    big_jumps = np.concatenate([neg_big_jumps, pos_big_jumps])

    return (X[:-1], small_jumps[:-1], big_jumps[:-1])

def plot(X, small_jumps, big_jumps):

    t = np.arange(len(X))

    df = pd.DataFrame({'X': X, 't': t, 'sj': small_jumps, 'bj': big_jumps})

    ax = plt.gca()
    
    df.plot(kind='line', x='t', y='sj', label='Brownian Motion', ax=ax)
    df.plot(kind='line', x='t', y='bj', label='Poisson Process', ax=ax)
    df.plot(kind='line', x='t', y='X', label='CGMY', ax=ax)
    
    plt.title('Simulation of CGMY process')
    plt.xlabel('t')
    plt.ylabel('CGMY')
    plt.show()

def main():
    np.random.seed(2019)  # set seed

    # C = 7
    # G = 20
    # M = 7
    # Y = 0.5

    C = 21.34
    G = 49.78
    M = 48.40
    Y = 0.0037

    D = 200
    sigma_squared = 0.5**2
    gamma = 2
    alpha = 0.2

    X, small_jumps, big_jumps = CGMY(C, G, M, Y, gamma, sigma_squared, D, alpha)

    plot(X, small_jumps, big_jumps)

if __name__ == "__main__":
    main()

