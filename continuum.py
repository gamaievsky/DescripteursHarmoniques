import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#PARAMETRES
shepard = True
timbre_def = (7,1/2,0.01)
K = timbre_def[0]
decr = timbre_def[1]
σ = timbre_def[2]
Σ = 3.0

#SPECTRE

def spectre(f0, K = K, decr = decr, σ = σ, shepard = shepard):
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


def energy(f0, K = K, decr = decr, σ = σ, shepard = shepard):
    return LA.norm(spectre(f0,K,decr,σ,shepard))**2

def concordance(f0,f1, K = K, decr = decr, σ = σ, shepard = shepard):
    return np.sum(spectre(f0,K,decr,σ,shepard)*spectre(f1,K,decr,σ,shepard))

# print(concordance(261, 361))

def courbe(param, ambitus = 12, f0 = 261, K = K, decr = decr, σ = σ, shepard = shepard):
    interv = np.arange(0,ambitus,0.05)
    C = np.zeros(np.shape(interv))
    if param == 'energy':
        for i,int in enumerate(interv):
            C[i] = energy((2.0**(int/12)) * f0,K,decr,σ,shepard)
    elif param == 'concordance':
        for i,int in enumerate(interv):
            C[i] = concordance(f0,2**(int/12) * f0,K,decr,σ,shepard)
    return C




# f, ax = plt.subplots(1)
# interv = np.arange(0,12,0.05)
# plt.plot(interv,C)
# plt.vlines(range(0, ambitus+1), 0, max(C), alpha=0.4, linestyle='--')
# plt.vlines(6, 0, max(C), color = 'r', alpha=0.9, linestyle='--')
# plt.xlabel('Intervalle en demi-tons')
# plt.ylabel(param)
# plt.title(param[0].upper() + param[1:] + ' courbe' + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# # ax.set_ylim(bottom=0)
# plt.show()

#######################
# Shepard vs non Shepard

param = 'concordance'
C1 = courbe('concordance', shepard = False)
C2 = courbe('concordance', shepard = True)
C1 = C1/max(C1)
C2 = C2/max(C2)

f, ax = plt.subplots(1)
interv = np.arange(0,12,0.05)
plt.plot(interv,C1, label = 'non Shepard')
plt.plot(interv,C2, label = 'Shepard')
plt.vlines(range(0, 12+1), 0, max(C1), alpha=0.4, linestyle='--')
plt.vlines(6, 0, max(C1), color = 'r', alpha=0.9, linestyle='--')
plt.xlabel('Intervalle en demi-tons')
plt.ylabel(param)
plt.title(param[0].upper() + param[1:] + ' courbe' + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# ax.set_ylim(bottom=0)
plt.legend(frameon=True, framealpha=0.75)
plt.show()


#######################
# Variation de K

# C1 = courbe('concordance', ambitus = 12, K = 3)
# C2 = courbe('concordance', ambitus = 12, K = 5)
# C3 = courbe('concordance', ambitus = 12, K = 11)
#
#
# f, ax = plt.subplots(1)
# param = 'concordance'
# interv = np.arange(0,12,0.05)
# plt.plot(interv,C1,label = 'K = 3')
# plt.plot(interv,C2,label = 'K = 5')
# plt.plot(interv,C3,label = 'K = 11')
# plt.vlines(range(0, 12+1), 0, max(C3), alpha=0.3, linestyle='--')
# plt.vlines(6, 0, max(C3), color = 'r', alpha=0.9, linestyle='--')
# plt.xlabel('Intervalle en demi-tons')
# plt.ylabel(param)
# plt.title(param[0].upper() + param[1:] + ' courbe' + '\nσ : {}, decr : {}'.format(σ,decr))
# # ax.set_ylim(bottom=0)
# plt.legend(frameon=True, framealpha=0.75)
# plt.show()
