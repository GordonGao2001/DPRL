import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# mu = 0.001, 0.015, 0.05
# v = 0.01, 0.005, 0.0025
# f = 500, 300, 200
c = 100
T = 600
value = []
optimal = []

def lamda_t(t, mu, v):
    p = mu*(np.exp(v*t))
    return p

def calculate_v():
    # value = [[0]*601]*101 
    value = [[0 for i in range(601)]for j in range(101)]
    policy = [[0 for i in range(600)]for j in range(100)]
    list_value = [[500, 300, 200],
                  [0.001, 0.015, 0.05],
                  [0.01, 0.005, 0.0025]]
    for x in range(1, c+1):
        for t in reversed(range(T)):
            vtx = []
            for l in range(len(list_value[0])):
                lamda = 0
                for i in range(l+1):
                    lamda += lamda_t(t+1, list_value[1][i], list_value[2][i])
                f = list_value[0][l]
                vx = lamda*(f+value[x-1][t+1])+(1-lamda)*value[x][t+1]
                vtx.append(vx)
            value[x][t] = max(vtx)
            index = vtx.index(max(vtx))
            policy[x-1][t] = list_value[0][index]
    return value, policy

value, policy = calculate_v()
print(value)
print(policy)
npvalue = np.array(value)
print(npvalue.shape)
nppolicy = np.array(policy)
print(nppolicy.shape)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(nppolicy)
ax.set_xlabel('Time', fontsize=7)
ax.set_ylabel('Capacity', fontsize=7)
plt.rc('xtick', labelsize=5) 
plt.rc('ytick', labelsize=5)
# plt.savefig('policy.png', bbox_inches='tight')
plt.show()