import os
import numpy as np
import matplotlib.pyplot as plt


σ = 0.01
Σ = 3.0
f0 = 630.1
K = 5
decr = 1/2

def repr_classe(f0,fmin,fmax):
    if fmin <= f0 < fmax: return f0
    elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
    elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
f0 = repr_classe(f0,261.0,522.0)




n = np.arange(0,16,0.001)
p0 = np.log2(261)
freq = np.exp(np.log(2)*n)
E = np.exp(-(n - p0)**2 / (2 * Σ**2))
norm = 0
S = np.zeros(np.shape(n))

for k in range(1,K+1):
    f = repr_classe(k*f0,261.0,522.0)
    p = np.log2(f)
    for i in range(-9,9):
        if 0 < p +i < 16:
            S += (1/k**decr) * np.exp(-(n - (p+i))**2 / (2 * σ**2))
    if np.log2(k).is_integer():
        norm += 1/k**decr

plot = plt.figure()
plt.xscale('log',basex = 2)
plt.plot(freq,norm*E, ls = '--')
plt.plot(freq,S*E)
plt.xlabel('Frequence')
plt.show()
