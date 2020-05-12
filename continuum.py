import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#PARAMETRES
shepard = True
timbre_def = (13,1/2,0.01)
K = timbre_def[0]
decr = timbre_def[1]
σ = timbre_def[2]
Σ = 3.0

#SPECTRE

def spectre(f0):
    nb_oct = 16
    n = np.arange(0,nb_oct,0.001)
    S = np.zeros(np.shape(n))

    if not shepard:
        for k in range(1, K+1):
            S += (1/k**decr) * np.exp(-(n - np.log2(k*f0))**2 / (2 * σ**2))

    else:
        def repr_classe(f0,fmin,fmax):
            if fmin <= f0 < fmax: return f0
            elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
            elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
        f0 = repr_classe(f0,261.0,522.0)
        p0 = np.log2(261)
        Σ = 3.0
        E = np.exp(-(n - p0)**2 / (2 * Σ**2))

        for k in range(1,K+1):
            f = repr_classe(k*f0,261.0,522.0)
            p = np.log2(f)
            for i in range(-10,10):
                if 0 < p+i < nb_oct:
                    S += (1/k**decr) * np.exp(-(n - (p+i))**2 / (2 * σ**2))
    return S


def energy(f0):
    return LA.norm(spectre(f0))**2

def concordance(f0,f1):
    return np.sum(spectre(f0)*spectre(f1))

# print(concordance(261, 361))

def courbe(param, ambitus = 12, f0 = 261):
    interv = np.arange(0,12,0.05)
    C = np.zeros(np.shape(interv))
    if param == 'energy':
        for i,int in enumerate(interv):
            C[i] = energy((2.0**int) * f0)
    elif param == 'concordance':
        for i,int in enumerate(interv):
            C[i] = concordance(f0,2**(int/12) * f0)

    f, ax = plt.subplots(1)
    plt.plot(interv,C)
    plt.vlines(range(1, ambitus), 0, max(C), alpha=0.4, linestyle='--')
    plt.vlines(6, 0, max(C), color = 'r', alpha=0.9, linestyle='--')
    plt.xlabel('Intervalle en demi-tons')
    plt.ylabel(param)
    plt.title('Courbe de ' + param + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
    ax.set_ylim(bottom=0)
    plt.show()

# def courbe(param, ambitus = 12, f0 = 261):
#     freq = np.arange(261,600,1)
#     E = np.zeros(np.shape(freq))
#     if param == 'energy':
#         for i,f in enumerate(freq):
#             E[i] = energy(f)
#     elif param == 'concordance':
#         for i,f in enumerate(freq):
#             E[i] = concordance(f0,f)
#     elif param == 'note_shepard':
#         def repr_classe(f0,fmin,fmax):
#             if fmin <= f0 < fmax: return f0
#             elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
#             elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
#         for i,f in enumerate(freq):
#             E[i] = repr_classe(f,261.0,522.0)
#
#     f, ax = plt.subplots(1)
#     plt.plot(freq, E)
#     plt.xlabel('Fréquence')
#     plt.ylabel(param)
#     # ax.set_ylim(bottom=0)
#     plt.show()


courbe('concordance')
