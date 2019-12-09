#!/usr/bin python

from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate
import numpy as np
import math

##                        
## 1. generate D steps of the brownian motion
##
def brownian_motion(D):
    dt = 1./D
    delta = D/5
    B = norm.rvs(size=D, scale=delta*math.sqrt(dt))
    B = np.insert(B[:-1], 0, 0)
    B = np.cumsum(B)
    return B

##                        
## 2. generate D steps of the a poisson process with parameter l (lambda)
##
def poisson_process(D, l):    
    dt = 1./D                              # compute time step
    exp = np.random.exponential(1./l, D)   # generate D random samples from Exp(1/l)
    exp = np.cumsum(exp)                   # sum them cumulatively
    exp = np.insert(exp[:-1], 0, 0)        # insert 0 element at the beginning of vector
                                           # and suppress last element
    N = np.zeros(D)                        # start with N = [0, ..., 0]

    LS = np.linspace(0, 1, num=D)          # create an evenly spaced vector in [0, 1]
                                           # with D elements

    for i in range(1, D): # i is for N[i]
        satisfy_condition = []
        for j in range(0, D): # j is for exp[j]
            if exp[j] <= LS[i]*dt:
                satisfy_condition.append(j)

        N[i] = max(satisfy_condition)

    return N

##                        
## 3. generate partitions using inverse linear boundaries technique
##
# alpha: lower and upper limits
# D: total number of points considered, where K = D/2
def LS_CGMY(alpha, D):
    K = int((D/2)+1)

    LS1 = np.zeros(K)
    LS2 = np.zeros(K)

    for i in range(1, K+1):
       LS1[i-1] =  -alpha/i
       LS2[K-i] = alpha/i

    return (LS1, LS2)

##                        
## 4. compute \lambda_i's
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
## 5. compute c_i's
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
        integral = compute_c_neg(C, G, M, Y, (neg_partitions[i], neg_partitions[i+1]))
        neg_cs.append(math.sqrt((1./neg_lambdas[i])*integral))

    for i in range(0, len(pos_partitions)-1):
        integral = compute_c_pos(C, G, M, Y, (pos_partitions[i], pos_partitions[i+1]))
        pos_cs.append(math.sqrt((1./pos_lambdas[i])*integral))

    return (np.array(neg_cs), np.array(pos_cs))

##                        
## 6. compute \sigma^2(\epsilon)
##
def compute_sigma_squared_f(C, G, M, Y, epsilon):
    neg_result, _ = integrate.quad(lambda x: (x**2)*C*np.exp(G*x)*(-x**(-1-Y)), 
                                   -epsilon, 0)
    pos_result, _ = integrate.quad(lambda x: (x**2)*C*np.exp(-M*x)*(x**(-1-Y)), 
                                   0, epsilon) 
    return neg_result + pos_result

##                        
## 7. compute gamma
##
def compute_gamma(C, G, M, Y):
    int1, _ = integrate.quad(lambda x: np.exp(-M*x)*(x**(-Y)), 0, 1)
    int2, _ = integrate.quad(lambda x: np.exp(G*x)*(abs(x)**(-Y)), -1, 0)
    return C*(int1-int2)

##                        
## 8. simulate trajectories
##    
def CGMY(C, G, M, Y, sigma_squared, D, alpha):
    
    K = int(D/2)
    LS = LS_CGMY(alpha, D)
    neg_partitions = LS[0]
    pos_partitions = LS[1]
    neg_lambdas, pos_lambdas = compute_lambdas(C, G, M, Y, LS)
    neg_cs, pos_cs = compute_cs(C, G, M, Y, LS, neg_lambdas, pos_lambdas)

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

    for i in range(0, K):
        p = np.cumsum(np.random.poisson(pos_lambdas[i], K))
        pos_ps.append(p)
    
    epsilon = pos_partitions[0]
    sigma_til = math.sqrt(sigma_squared + compute_sigma_squared_f(C, G, M, Y, epsilon))
    gamma = compute_gamma(C, G, M, Y)

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

    return (X, small_jumps, big_jumps)

##                        
## 9. compute option price
##
def option_pricing(C, G, M, Y, D, sigma_squared, alpha, r, q, strike_prices, closing_prices, S0, N):
    
    m_new = r - q - C*gamma.rvs(Y)*(((M-1)**Y)-(M**Y)+((G+1)**Y)-(G**Y))

    T = (12-4)/12 + (30-18)/365.25

    for i in range(0, len(strike_prices)):
        option_prices = 0
        for j in range(0, N):
            X, _, _ = CGMY(C, G, M, Y, sigma_squared, D, alpha)
            S_T = S0*np.exp(m_new*T + X[-1])
            option_price = np.exp(-r*T)*max(S_T - strike_prices[i], 0)
            option_prices += option_price
        
        result = option_prices/N
        rmse = math.sqrt((closing_prices[i] - result)**2)
        print("----------------------------")
        print("K = " + str(strike_prices[i]))
        print("Simulated value = " + str(result))
        print("Closing price = " + str(closing_prices[i]))
        print("RMSE = " + str(rmse))

def plot(X, small_jumps, big_jumps):

    t = np.arange(len(X))

    df = pd.DataFrame({'X': X, 't': t, 'sj': small_jumps, 'bj': big_jumps})

    ax = plt.gca()
    
    df.plot(kind='line', x='t', y='sj', label='Brownian Motion', ax=ax)
    df.plot(kind='line', x='t', y='bj', label='Poisson Process', ax=ax)
    df.plot(kind='line', x='t', y='X', label='CGMY', ax=ax)
    
    plt.title('Simulation of CGMY process')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.show()

def plot_option_prices():
    strike_prices = [975, 1025, 1075, 1125, 1175, 1225, 1275]
    closing_prices = [173.30, 133.10, 97.60, 66.90, 42.50, 24.90, 13.20]
    simulated_prices = [177.32, 132.77, 106.74, 88.56, 78.07, 90.6, 77.66]

    df = pd.DataFrame({'K': strike_prices, 'cp': closing_prices, 'sp': simulated_prices})
    
    ax = plt.gca()

    df.plot(kind='line', x='K', y='sp', label='Model price', ax=ax)
    df.plot(kind='line', x='K', y='cp', label='Market price', ax=ax)
    
    plt.title('S&P 500 Call Option Prices')
    plt.xlabel('K')
    plt.ylabel('Option price')
    plt.show()
    

def main():
    # np.random.seed(2019)  # set seed

    # C = 7
    # G = 20
    # M = 7
    # Y = 0.5

    # Plot random trajectories
    C = 21.34
    G = 49.78
    M = 48.40
    Y = 0.0037
    D = 400
    sigma_squared = 0.5**2
    alpha = 0.2
    X, small_jumps, big_jumps = CGMY(C, G, M, Y, sigma_squared, D, alpha)
    plot(X, small_jumps, big_jumps)

    # data from Schoutens (Levy Processes in Finance: Pricing Financial Derivatives)
    # C = 0.0244
    # G = 0.0765
    # M = 30.7515
    # Y = 0.5
    # D = 200
    # sigma_squared = 0.0001**2
    # alpha = 0.2
    # r = 0.019
    # q = 0.012
    # strike_prices = [975, 1025, 1075, 1125, 1175, 1225, 1275]
    # closing_prices = [173.30, 133.10, 97.60, 66.90, 42.50, 24.90, 13.20]
    # N = 5000
    # S0 = 1124.47
    # option_pricing(C, G, M, Y, D, sigma_squared, alpha, r, q, strike_prices, closing_prices, S0, N)

    # plot_option_prices()

if __name__ == "__main__":
    main()

