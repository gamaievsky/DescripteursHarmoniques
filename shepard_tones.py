import os
import numpy as np
import matplotlib.pyplot as plt

# f0 = 440.0
#
# σ = 0.03
# Σ = 3.0
# K = 5
# decr = 1/2
#
# def repr_classe(f0,fmin,fmax):
#     if fmin <= f0 < fmax: return f0
#     elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
#     elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
# f0 = repr_classe(f0,261.0,522.0)
#
#
#
#
# n = np.arange(0,16,0.001)
# p0 = np.log2(261)
# # f0 = 440
# freq = np.exp(np.log(2)*n)
#
# E = np.exp(-(n - p0)**2 / (2 * Σ**2))
# norm = 0
# S = np.zeros(np.shape(n))
#
# for k in range(1,K+1):
#     f = repr_classe(k*f0,261.0,522.0)
#     p = np.log2(f)
#     for i in range(-9,9):
#         if 0 < p +i < 16:
#             S += (1/k**decr) * np.exp(-(n - (p+i))**2 / (2 * σ**2))
#     if np.log2(k).is_integer():
#         norm += 1/k**decr
#
# plot = plt.figure()
# plt.xscale('log',basex = 2)
# plt.plot(freq,norm*E, ls = '--')
# plt.plot(freq,S*E, label = 'Pitch-class {}'.format(f0))
# plt.xlabel('Frequence')
# plt.title('Spectre de Shepard' + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# plt.legend(frameon=True, framealpha=0.75)
# plt.show()


#######################


f0 = 440.0

Σ = 3.0
K, decr = 5, 0.5
n = np.arange(0,16,0.001)
p0 = np.log2(261)
E = np.exp(-(n - p0)**2 / (2 * Σ**2))
# f0 = 440
freq = np.exp(np.log(2)*n)



def spectre_pic(f0):

    dic_spectre = {}
    def repr_classe(f0,fmin,fmax):
        if fmin <= f0 < fmax: return f0
        elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
        elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
    f = repr_classe(f0,261.0,522.0)
    p0 = np.log2(261)
    Σ = 3.0
    
    # Construction de dic_spectre
    for k in range(1,K+1):
        f = repr_classe(k*f0,261.0,522.0)
        p = np.log2(f)
        for i in range(-8,8):
            if 0 < p + i < 16:
                fp = f*2**i
                if fp in dic_spectre:
                    dic_spectre[fp] += (1/k**decr)* np.exp(-(np.log2(fp) - p0)**2 / (2 * Σ**2))
                else : dic_spectre[fp] = (1/k**decr)* np.exp(-(np.log2(fp) - p0)**2 / (2 * Σ**2))

    return dic_spectre

norm = 0
for k in range(1,K+1):
    if np.log2(k).is_integer():
        norm += 1/k**decr

dic_spectre = spectre_pic(f0)
# print(dic_spectre)

plot = plt.figure()
plt.xscale('log',basex = 2)
plt.plot(freq,norm*E, ls = '--')
leg = False
for f in dic_spectre:
    if not leg:
        plt.vlines(f, 0, dic_spectre[f], alpha=0.9, label = 'Pitch_class {}'.format(f0))
        leg = True
    else: plt.vlines(f, 0, dic_spectre[f], alpha=0.9)
# plt.plot(freq,S*E)
plt.xlabel('Frequence')
plt.title('Spectre de Shepard en pics' + '\nK : {}, decr : {}'.format(K,decr))
plt.legend(frameon=True, framealpha=0.75)
plt.show()
