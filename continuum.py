import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from numpy import linalg as LA
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import itertools
import pickle
import parametres
from enumeration import Enumerate, Normal_Reduction, Interval_Reduction

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


#PARAMETRES
shepard = True
timbre_def = (11,0.5,0.005)
K = timbre_def[0]
decr = timbre_def[1]
σ = timbre_def[2]
Σ = 2.0 # en octaves
ε = 0.05

#SPECTRE

def spectre(f0, K = K, decr = decr, σ = σ, shepard = shepard, Σ = Σ, plot = False, plot_pics = False, return_pics = False):
    nb_oct = 16
    n = np.arange(0,nb_oct,0.001)
    # n = np.arange(8,13,0.001)
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
        p0 = np.log2(261.0)
        E = np.exp(-(n - p0)**2 / (2 * Σ**2))

        for k in range(1,K+1):
            f = repr_classe(k*f0,261.0,522.0)
            p = np.log2(f)
            for i in range(-10,10):
                if 0 < p+i < nb_oct:
                    S += (1/k**decr) * np.exp(-(n - (p+i))**2 / (2 * σ**2))
        max_val = np.max(S)
        S *= E

        # Construction de dic_spectre
        dic_spectre = {}
        for k in range(1,K+1):
            f = repr_classe(k*f0,261.0,522.0)
            p = np.log2(f)
            for i in range(-8,8):
                if 0 < p + i < 16:
                    fp = f*2**i
                    if fp in dic_spectre:
                        dic_spectre[fp] += (1/k**decr)* np.exp(-(np.log2(fp) - p0)**2 / (2 * Σ**2))
                    else : dic_spectre[fp] = (1/k**decr)* np.exp(-(np.log2(fp) - p0)**2 / (2 * Σ**2))
    if plot:
        freq = np.exp(np.log(2)*n)
        if not shepard:
            env = (f0**decr) * np.power(freq,-decr)
            f, ax = plt.subplots(1)
            # ax.set_ylim(top = 1)
            plt.xscale('log',basex = 2)
            plt.plot(freq[600:],env[600:], ls = '--')
            plt.plot(freq,S, label = 'Pitch {} Hz'.format(f0))
            plt.xlabel('Frequence (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Spectre en gaussiennes' + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
        else:
            plt.xscale('log',basex = 2)
            plt.plot(freq, E*max_val, ls = '--')
            plt.plot(freq, S, label = 'Pitch-class La')
            plt.xlabel('Frequence (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Spectre de Shepard' + '\nK : {}, σ : {}, decr : {}, Σ : {}'.format(K,σ,decr,Σ))
        plt.legend(frameon=True, framealpha=0.75)
        plt.show()
    if plot_pics:
        freq = np.exp(np.log(2)*n)
        if not shepard:
            env = (f0**decr) * np.power(freq,-decr)
            f, ax = plt.subplots(1)
            # ax.set_ylim(top = 1)
            plt.xscale('log',basex = 2)
            plt.plot(freq[600:],env[600:], ls = '--')
            plt.vlines([k*f0 for k in range(1,K+1)],0,[1/k**decr for k in range(1,K+1)], label = 'Pitch {} Hz'.format(f0))
            plt.xlabel('Frequence (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Spectre en pics' + '\nK : {}, decr : {}'.format(K,decr))
        else:
            plt.xscale('log',basex = 2)
            plt.plot(freq, E*max_val, ls = '--')
            plt.vlines([f for f in dic_spectre],0,[dic_spectre[fp] for fp in dic_spectre], label = 'Pitch-class La')
            plt.xlabel('Frequence (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Spectre de Shepard en pics' + '\nK : {}, decr : {}, Σ : {}'.format(K,decr,Σ))

        plt.legend(frameon=True, framealpha=0.75)
        plt.show()

    if return_pics: return dic_spectre
    else: return S

def energy(f0,  K = K, decr = decr, σ = σ, shepard = shepard):
    return LA.norm(spectre(f0, K = K, decr = decr, σ = σ, shepard = shepard))**2

def concordance(f0,f1, K = K, decr = decr, σ = σ, shepard = shepard):
    return np.sum(spectre(f0,K,decr,σ,shepard)*spectre(f1,K,decr,σ,shepard))

def roughness(f0,f1, K = K, decr = decr, σ = σ, shepard = shepard):
    def rough(f0,f1):
        s = 0.24/(0.021*f0 + 19)
        return np.exp(-parametres.β1*s*(f1-f0))-np.exp(-parametres.β2*s*(f1-f0))
    R = 0
    if not shepard:
        for i in range(K):
            for j in range(K):
                freqs = sorted([(i+1)*f0,(j+1)*f1])
                R += (1.0/((i+1)*(j+1))**decr) * rough(freqs[0],freqs[1])
    else:
        dic0 = spectre(f0, return_pics = True)
        dic1 = spectre(f1, return_pics = True)
        for fr0 in dic0:
            for fr1 in dic1:
                freqs = sorted([fr0,fr1])
                R += (dic0[fr0]*dic1[fr1]) * rough(freqs[0],freqs[1])
    return R

def conc3(f0,f1,f2, K = K, decr = decr, σ = σ, shepard = shepard):
    return np.sum(spectre(f0,K,decr,σ,shepard)*spectre(f1,K,decr,σ,shepard)*spectre(f2,K,decr,σ,shepard))

def tension(f0,f1,f2, K = K, decr = decr, σ = σ, shepard = shepard):
    def tens(f0,f1,f2):
        x = np.log2(f1/f0)
        y = np.log2(f2/f1)
        if 12*(abs(y-x)) < 2:
                return  np.exp(-(12*(abs(y-x))/parametres.δ)**2)
        else: return 0
    T = 0
    if not shepard:
        for i in range(K):
            for j in range(K):
                for k in range(K):
                    freqs = sorted([(i+1)*f0,(j+1)*f1,(k+1)*f2])
                    T += (1.0/((i+1)*(j+1)*(k+1))**decr) * tens(freqs[0],freqs[1],freqs[2])
    else:
        dic0 = spectre(f0, return_pics = True)
        dic1 = spectre(f1, return_pics = True)
        dic2 = spectre(f2, return_pics = True)
        for fr0 in dic0:
            for fr1 in dic1:
                for fr2 in dic2:
                    freqs = sorted([fr0,fr1,fr2])
                    T += (dic0[fr0]*dic1[fr1]*dic2[fr2]) * tens(freqs[0],freqs[1],freqs[2])
    return T

def harmonicity(spectrum, f0 = 65.0, K = K, decr = decr, σ = σ, shepard = shepard):
    corr = np.correlate(spectrum, spectre(f0, K = K + 10, decr = 0, σ = σ, shepard = shepard), 'full')
    return np.nanmax(corr)

def DiffConcordance(freqs1,freqs2,K = K, decr = decr, σ = σ, shepard = shepard):
    S1 = spectre(freqs1[0])
    S2 = spectre(freqs2[0])
    for f in freqs1[1:]:
        S1 += spectre(f)
    for f in freqs2[1:]:
        S2 += spectre(f)
    return np.sum(S1*S2) / (LA.norm(S1)*LA.norm(S2))

def DiffRoughness(freqs1,freqs2, K = K, decr = decr, σ = σ, shepard = shepard):
    S1 = spectre(freqs1[0])
    S2 = spectre(freqs2[0])
    for f in freqs1[1:]:
        S1 += spectre(f)
    for f in freqs2[1:]:
        S2 += spectre(f)
    R = 0.
    for f1 in freqs1:
        for f2 in freqs2:
            freqs = sorted([f1,f2])
            R += roughness(freqs[0],freqs[1])
    return R / (LA.norm(S1)*LA.norm(S2))

def HarmonicChange(freqs1,freqs2, K = K, decr = decr, σ = σ, shepard = shepard):
    S1 = spectre(freqs1[0])
    S2 = spectre(freqs2[0])
    for f in freqs1[1:]:
        S1 += spectre(f)
    for f in freqs2[1:]:
        S2 += spectre(f)
    return np.sum(np.abs(S2-S1)) / np.sqrt((LA.norm(S1)*LA.norm(S2)))


def Courbe(param, ambitus = 12, f0 = 261, K = K, decr = decr, σ = σ, Σ = Σ, shepard = shepard):
    interv = np.arange(0,ambitus+ε,ε)
    C = np.zeros(np.shape(interv))
    if param == 'energy':
        for i,int in enumerate(interv):
            C[i] = LA.norm(spectre(f0*2**(int/12.0), Σ = Σ),shepard = shepard)**2
    elif param == 'concordance':
        for i,int in enumerate(interv):
            C[i] = concordance(f0,2**(int/12.0) * f0, K = K, decr = decr, σ = σ, shepard = shepard)
            C[i] /= LA.norm(spectre(f0,shepard = shepard) + spectre(2**(int/12.0) * f0,shepard = shepard))**2
    elif param == 'roughness':
        for i,int in enumerate(interv):
            C[i] = roughness(f0,2**(int/12.0) * f0, K = K, decr = decr, σ = σ, shepard = shepard)
            C[i] /= LA.norm(spectre(f0,shepard = shepard) + spectre(2**(int/12.0) * f0,shepard = shepard))**2
    # elif param == 'tension':
    #     for i,int in enumerate(interv):
    elif param == 'harmonicity':
        for i,int in enumerate(interv):
            C[i] = harmonicity(spectre(f0,shepard = shepard) + spectre(f0 * 2**(int/12.0),shepard = shepard))
            C[i] /= LA.norm(spectre(f0,shepard = shepard) + spectre(2**(int/12.0) * f0,shepard = shepard))
    return C


def Carte(param, ambitus = 12, f0 = 261, K = K, decr = decr, σ = σ, shepard = shepard, classe = None):
    if param in ['concordance']:
        courbe = Courbe(param, ambitus = 2*ambitus, f0 = f0, K = K, decr = decr, σ = σ, shepard = shepard)
    interv = np.arange(0,ambitus+ε,ε)
    N = len(interv)
    G = np.zeros((N,N))
    for i,int1 in enumerate(interv):
        for j,int2 in enumerate(interv):
            if classe == None:
                if param == 'concordance':
                    G[j,i] = courbe[i] + courbe[j] + courbe[i+j]
                    G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)))**2
                if param == 'roughness':
                    G[j,i] = roughness(f0,f0*2**(int1/12.0), K = K, decr = decr, σ = σ, shepard = shepard) + roughness(f0,f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard) + roughness(f0*2**(int1/12.0), f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard)
                    G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)))**2
                if param == 'conc3':
                    G[j,i] = conc3(f0,f0*2**(int1/12.0),f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard)
                    G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)),3)**3
                if param == 'tension':
                    G[i,j] = tension(f0,f0*2**(int1/12.0),f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard)
                if param == 'harmonicity':
                    G[j,i] = harmonicity(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)), K = K, decr = decr, σ = σ, shepard = shepard)
                    G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)))
            elif classe == 'prime':
                δ = 0.1
                if (int1-δ<=int2<=6+δ-int1/2.):
                    if param == 'concordance':
                        G[j,i] = courbe[i] + courbe[j] + courbe[i+j]
                        G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)))**2
                    if param == 'roughness':
                        G[j,i] = roughness(f0,f0*2**(int1/12.0), K = K, decr = decr, σ = σ, shepard = shepard) + roughness(f0,f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard) + roughness(f0*2**(int1/12.0), f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard)
                        G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)))**2
            elif classe == 'normal':
                δ = 0.1
                if (int2<=6+δ-int1/2.) and (int2<=12+2*δ-2*int1):
                    if param == 'conc3':
                        G[j,i] = conc3(f0,f0*2**(int1/12.0),f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard)
                        G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)),3)**3
                    if param == 'tension':
                        G[i,j] = tension(f0,f0*2**(int1/12.0),f0*2**((int1+int2)/12.0), K = K, decr = decr, σ = σ, shepard = shepard)
                    if param == 'harmonicity':
                        G[j,i] = harmonicity(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)), K = K, decr = decr, σ = σ, shepard = shepard)
                        G[j,i] /= LA.norm(spectre(f0) + spectre(f0*2**(int1/12.0)) + spectre(f0*2**((int1+int2)/12.0)))

    if classe == 'prime':
        m = np.max(G)
        for i,int1 in enumerate(interv):
            for j,int2 in enumerate(interv):
                if (int1-δ>=int2) or (int2>=6+δ-int1/2.):
                    G[j,i] = m
    if classe == 'normal':
        m = np.max(G)
        for i,int1 in enumerate(interv):
            for j,int2 in enumerate(interv):
                if (int2>=6+δ-int1/2.) or (int2>=12+2*δ-2*int1):
                    G[j,i] = m


    return G

########################

# Classes des accords à 4 sons

with open('Dic_iv.pkl', 'rb') as f:
    Dic_iv = pickle.load(f)
with open('Dic_Harm.pkl', 'rb') as f:
    Dic_Harm = pickle.load(f)

ambitus = 6
ε = 0.1
interv = np.arange(0,ambitus+ε,ε)

liste_plans_prime = [(1,2,1,-12), (1,1,2,-12), (1,-1,0,0), (0,1,-1,0), (1,0,0,0), (0,1,0,0), (0,0,1,-6)]
liste_plans_normal = [(2,1,1,-12), (1,2,1,-12), (1,1,2,-12),(1,0,0,0),(0,1,0,0),(0,0,1,0)]

def Intersection(liste_plans, d):
    # Fonction qui repère les points d'intersection entre plusieurs plans, de multyiplicité d
    liste_inter3 = []
    for x in interv:
        for y in interv:
            for z in interv:
                num = 0
                for plan in liste_plans:
                    if (plan[0]*x + plan[1]*y + plan[2]*z + plan[3] ==0):
                        num+=1
                if num >= d:
                    liste_inter3.append((x,y,z))
    return liste_inter3

def Plane(ax,a,b,c,d):
    # Fonction qui trace un plan
    if c != 0:
        xx, yy = np.meshgrid(interv, interv)
        zz = (-a * xx - b * yy - d) * 1. /c
    else:
        yy,zz = np.meshgrid(interv, interv)
        xx = (- b * yy - d) * 1. /a
    ax.plot_surface(xx,yy,zz,alpha=0.3)

def DiscreteChords(ax, M = 12, N=4, classe = 'prime', descr = None, sp = (11,0.5,0.005)):
    # Fonction qui trace toutes les classes premières ou normales dans une subdivision M donnée
    if classe == 'prime':
        liste_acc = Interval_Reduction(Enumerate(N,M-1),M)
    else:
        liste_acc = Normal_Reduction(Enumerate(N,M-1),M)
    liste_acc_red = [[((a+[M])[i+1] - (a+[M])[i])*12./M for i in range(len(a))] for a in liste_acc]
    if descr != None:
        liste_descr = [Dic_Harm[(sp[0],sp[1],sp[2])][descr][num] for num in range(20,35)]
        normalize = colors.Normalize(vmin=min(liste_descr), vmax=max(liste_descr))
        ax.scatter([acc[0] for acc in liste_acc_red],[acc[1] for acc in liste_acc_red],[acc[2] for acc in liste_acc_red], c = liste_descr, cmap=cm.jet, norm=normalize, edgecolors='k',alpha = 1.,s=70)
        # cb = fig.colorbar(cm.ScalarMappable(norm=normalize, cmap=cm.jet), ax=ax, orientation = 'horizontal')
        # cb.ax.set_title('Concordance')
    else:
        for acc in liste_acc_red:
            ax.scatter(acc[0],acc[1],acc[2], color = 'k')

def Traverse(p0,p1,plan, plot = False, ax = None, color = 'k'):
    # Point d'intersection entre une droite et un plan (si tout va bien...)
    if plan[2]!=0:
        a = np.array([[plan[0],plan[1],plan[2]], [p1[2]-p0[2], 0, -(p1[0]-p0[0])], [0, p1[2]-p0[2], -(p1[1]-p0[1])]])
        b = np.array([-plan[3], p0[0]*p1[2] - p0[2]*p1[0], p0[1]*p1[2] - p0[2]*p1[1]])
    else:
        a = np.array([[plan[0],plan[1],plan[2]], [-(p1[1]-p0[1]), p1[0]-p0[0], 0], [-(p1[2]-p0[2]), 0, p1[0]-p0[0]]])
        b = np.array([-plan[3], p0[1]*p1[0] - p0[0]*p1[1], p0[2]*p1[0] - p0[0]*p1[2]])

    sol = np.linalg.solve(a, b)
    if plot:
        ax.scatter(sol[0],sol[1],sol[2],marker = '*',color = color, alpha = 1)
    return sol


def Normal(ax, form, color = 'k'):
    def cyclic_perm(a):
        n = len(a)
        b = [[a[i - j] for i in range(n)] for j in range(n)]
        return b
    l = cyclic_perm(form)
    p = liste_plans_normal
    liste_off = []
    # Tracer les points
    for acc in l:
        if (acc[0]*p[0][0] + acc[1]*p[0][1] + (acc[2]+p[0][3])*p[0][2] <= 0) and (acc[0]*p[1][0] + acc[1]*p[1][1] + (acc[2]+p[1][3])*p[1][2] <= 0) and (acc[0]*p[2][0] + acc[1]*p[2][1] + (acc[2]+p[2][3])*p[2][2] <= 0):
            ax.scatter(acc[0],acc[1],acc[2],color=color,marker = 'o',alpha = 0.7)
            acc_in = acc
        else :
            ax.scatter(acc[0],acc[1],acc[2],color=color)
            liste_off.append(acc)
    # tracer les lignes
    if len(liste_off)>0:
        liste_off.sort(reverse = True)
        for i,acc_off in enumerate(liste_off):
            inters = Traverse(acc_in,acc_off,liste_plans_normal[i], plot = True, ax = ax, color = color)
            ax.plot([acc_in[0], inters[0]], [acc_in[1], inters[1]], [acc_in[2], inters[2]], linestyle=':', color = color, alpha = 0.7)
            ax.plot([acc_off[0], inters[0]], [acc_off[1], inters[1]], [acc_off[2], inters[2]], linestyle='-', color = color)

        ax.plot([acc[0] for acc in liste_off+[liste_off[0]]], [acc[1] for acc in liste_off+[liste_off[0]]], [acc[2] for acc in liste_off+[liste_off[0]]], linestyle = '-',color = color)


def Prime(ax, form, color = 'k', descr = None, sp = (11,0.5,0.005)):
    l_normal = list(itertools.permutations([form[0], form[1], form[2]]))
    acc_in = form
    for acc_off in [l_normal[2]]:#, l_normal[4], l_normal[5]]:
        inters = Traverse(acc_in, acc_off, (1,-1,0,0), plot = True, ax = ax, color = color)
        ax.plot([acc_in[0], inters[0]], [acc_in[1], inters[1]], [acc_in[2], inters[2]], linestyle=':', color = color, alpha = 0.7)
        ax.plot([acc_off[0], inters[0]], [acc_off[1], inters[1]], [acc_off[2], inters[2]], linestyle='-', color = color)
    for acc_off in [l_normal[1]]:#, l_normal[3]]:
        inters = Traverse(acc_in, acc_off, (0,1,-1,0), plot = True, ax = ax, color = color)
        ax.plot([acc_in[0], inters[0]], [acc_in[1], inters[1]], [acc_in[2], inters[2]], linestyle=':', color = color, alpha = 0.7)
        ax.plot([acc_off[0], inters[0]], [acc_off[1], inters[1]], [acc_off[2], inters[2]], linestyle='-', color = color)

    l_ordre = [l_normal[1],l_normal[3],l_normal[5],l_normal[4],l_normal[2]]
    ax.plot([acc[0] for acc in l_ordre], [acc[1] for acc in l_ordre], [acc[2] for acc in l_ordre], color = color, linestyle = '-', alpha = 1)

    if descr != None:
        liste_id = ['27-{}'.format(i) for i in range(1,7)]
        liste_acc = [Dic_iv[id] for id in liste_id]
        liste_descr = [Dic_Harm[(sp[0],sp[1],sp[2])][descr][id] for id in liste_id]
        normalize = colors.Normalize(vmin=min(liste_descr), vmax=max(liste_descr))
        ax.scatter([acc[0] for acc in liste_acc],[acc[1] for acc in liste_acc],[acc[2] for acc in liste_acc], c = liste_descr, cmap=cm.jet, norm=normalize, edgecolors='k',alpha = 1.,s=70)
        cb = fig.colorbar(cm.ScalarMappable(norm=normalize, cmap=cm.jet), ax=ax, orientation = 'horizontal')
        cb.ax.set_title('Tension')
    else:
        for acc in l_normal:
            ax.scatter(acc[0],acc[1],acc[2],color = color, alpha = 1)


def ContinuumDiscretum(ax, classe = 'prime', discret = False, M = 12, print_classes = False, alpha = 0.3, descr = None):



    # Représentation du volume
    l = Intersection(liste_plans_prime,4)
    ln = Intersection(liste_plans_normal,3)
    if classe =='prime':
        fc = ["C0","C1","C2","C4"]
        verts = [[l[0],l[1],l[2]], [l[0],l[2],l[3]], [l[1],l[2],l[3]], [l[0],l[1],l[3]]]
    else:
        fc = ["C0","C1","C2","C4","C3","C4"]
        verts = [[ln[1],ln[2],ln[4],ln[5]], [ln[2],ln[4],ln[6],ln[3]], [ln[5],ln[4],ln[6],ln[7]], [ln[0],ln[1],ln[2],ln[3]], [ln[0],ln[1],ln[5],ln[7]], [ln[0],ln[3],ln[6],ln[7]]]
    pc = Poly3DCollection(verts, facecolors=fc, linewidths=1, edgecolor = 'k', lw = 0.7,alpha = alpha)
    # ax.plot([acc[0] for acc in l+[l[0],l[1],l[3],l[0],l[2]]], [acc[1] for acc in l+[l[0],l[1],l[3],l[0],l[2]]], [acc[2] for acc in l+[l[0],l[1],l[3],l[0],l[2]]], c='k')
    tri = ax.add_collection3d(pc)

    if discret: DiscreteChords(ax, M = M, classe = classe, descr = descr)

    if print_classes:
        Normal(ax,[1,2,4,5], color = 'b')
        # Prime(ax,[1,2,4,5], color = 'b', descr = descr)


    ax.set_xlim(0,ambitus)
    ax.set_ylim(ambitus,0)
    ax.set_zlim(0,ambitus)
    plt.xlabel('Intervalle inférieur')
    plt.ylabel('Intervalle médian')
    ax.set_zlabel('Intervalle supérieur')
    # ax.zaxis.set_visible(False)
    # plt.title(param[0].upper() + param[1:] + ' des triades'+'\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))


fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')

# ContinuumDiscretum(ax, classe = 'prime', alpha = 0.4)
# ContinuumDiscretum(ax, classe = 'normal', print_classes = False, alpha = 0.3,discret = True, M = 2*12)
ContinuumDiscretum(ax, classe = 'normal', print_classes = True, alpha = 0.3)


plt.show()


#######################
# Affichage du spectre

# f0 = 440
# S = spectre(f0, plot_pics = True)
#######################

# Carte des classes premières et normales

# param = 'harmonicity'
# classe = 'normal'
# f0 = 261
# ambitus = 6
# plotlines = True
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
# # C = Carte(param, ambitus = ambitus,f0 = f0, classe = classe)
# # np.save('Classe_'+ param +'_K{}_sig{}_eps{}'.format(K,σ,ε),C)
# with open('Classe_'+ param +'_K{}_sig{}_eps{}.npy'.format(K,σ,ε), 'rb') as f:
#     C = np.load(f)
#
# fig, ax = plt.subplots(1,figsize=(9, 7))
# cs = ax.contourf(interv,interv,C,300,cmap=cm.jet)
# if plotlines:
#     plt.vlines(np.arange(0, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
#     plt.hlines(np.arange(0, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
# plt.plot([0,6],[0,6], color = 'k',alpha=0.7, linestyle=':', linewidth = 1.0)
#
# if classe == 'prime':
#     plt.plot([0,4],[6,4],[0,4],[0,4],color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0)
#     ax.fill_between([0,4,6],[0,0,0],[0,4,6],facecolor='w', interpolate=True, alpha = 1.0)
#     ax.fill_between([0,4,6],[6,4,6],[6,6,6],facecolor='w', interpolate=True, alpha = 1.0)
# if classe == 'normal':
#     plt.plot([0,4],[6,4], color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0)
#     plt.plot([4,6],[4,0], color = 'k',alpha=0.7, linestyle=':', linewidth = 1.0)
#     ax.fill_between([0,4,6],[6,4,0],[6,6,6],facecolor='w', interpolate=True, alpha = 1.0)
# plt.xlabel('Intervalle inférieur (en demi-tons)')
# plt.ylabel('Intervalle supérieur (en demi-tons)')
# # plt.title('Concordance d\'ordre 3 des triades'+'\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# # plt.title(param[0].upper() + param[1:] + ' des classes premières de triades'+'\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# plt.title('Harmonicité des classes normales de triades'+'\nK : {}, decr : {}'.format(K,decr))
# cbar = fig.colorbar(cs, format='%.1d')
# plt.show()

#######################
# Coeff Variation

# sig_tab = np.arange(0.5,3.5,0.1)
# l = []
# for sig in sig_tab:
#     C = Courbe('energy', ambitus = 12, Σ = sig)
#     var = np.std(C)/np.mean(C)
#     l.append(var)
# np.save('Table_var_shep_{}'.format(σ),l)
# # with open('Table_var_shep_{}.npy'.format(σ),'rb') as f:
# #     l = np.load(f)
#
# f, ax = plt.subplots(1)
# plt.plot(sig_tab,l)
# ax.set_yscale('log')
# plt.xlabel('Écart-type de l\'enveloppe du spectre (en octaves)')
# # plt.title('Écart-type de l\'enveloppe du spectre')
# plt.ylabel('Coefficient de variation de l\'énergie sur l\'octave')
# plt.show()


    # print('Σ = {} - Coefficient de variation : {}'.format(sig,var))


#######################
# Courbe de paramètre

# param = 'roughness'
# ambitus = 6
# C = Courbe(param, ambitus = ambitus)
# interv = np.arange(0,ambitus+ε,ε)
# # print('Variation : {}'.format(np.std(C)/np.mean(C)))
#
# f, ax = plt.subplots(1,figsize=(10, 4))
# plt.plot(interv,C, color = 'r',label = 'Rugosité')
# plt.vlines(range(0, ambitus+1), 0, max(C), alpha=0.4, linestyle='--')
# # plt.vlines([6], 0, max(C), color = 'r', alpha=0.9, linestyle='--')
# plt.xlabel('Classe d\'intervalles (en classes de demi-ton)')
# plt.ylabel('Rugosité de classe')
# # plt.title('Courbe de concordance, harmonicité et rugosité de classe' + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# # ax.set_yscale('log')
# # ax.set_ylim(bottom=0)
# # ax.set_xlim(0,24)
# plt.legend(frameon=True, framealpha=0.75)
# plt.show()




#######################
# Shepard vs non Shepard

# param = 'concordance'
# ambitus = 12
# C1 = Courbe('concordance',ambitus = ambitus, shepard = False)
# C2 = Courbe('concordance', ambitus = ambitus, shepard = True)
# # C1 = C1/max(C1)
# # C2 = C2/max(C2)
#
#
#
# f, ax = plt.subplots(1,figsize=(11, 5))
# interv = np.arange(0,ambitus+ε,ε)
# plt.plot(interv,C1, label = 'non Shepard')
# plt.plot(interv,C2, label = 'Shepard')
# plt.vlines(range(0, ambitus+1), 0, max(C1), alpha=0.4, linestyle='--')
# plt.vlines(6, 0, max(C2), color = 'r', alpha=0.9, linestyle='--')
# plt.xlabel('Intervalle en demi-tons')
# plt.ylabel(param)
# plt.title('Courbe de ' + param + '\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# # ax.set_ylim(bottom=0)
# plt.legend(frameon=True, framealpha=0.75)
# plt.show()



#######################
# Ajout d'une note

# param = 'ConcordanceTot'
# f0 = 261.0
# ambitus = 12
# plotlines = True
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
# N = len(interv)
# interv2 = np.concatenate([np.arange(-ambitus,0,ε),interv])
#
# C = Courbe('concordance', ambitus = ambitus,f0 = f0)
# M = Carte('conc3', ambitus = ambitus,f0 = f0)
# G = np.zeros((2*N - 1, N))
#
# C = 4*C
# M = 9*M
#
# for i in range(N):
#     for j in range(N-1,2*N-1):
#         G[j,i] = M[j-N-1,i]-C[i]
#     for j in range(N-1):
#         int_j = N-1-j
#         if interv[int_j]<interv[i]: G[j,i] = M[int_j,i-int_j]-C[i]
#         else: G[j,i] = M[i,int_j-i]-C[i]
#
#
# # np.save('G_concTot',G)
#
# # with open('G_harmonicity.npy','rb') as f:
# #     G = np.load(f)
#
# G = -G
#
# fig, ax = plt.subplots(1,figsize=(5, 7.5))
# # cs = ax.contourf(interv,interv,B,2,cmap=cm.seismic)
# midpoint = 1 - np.max(G) / (np.max(G) + abs(np.min(G)))
# shifted_cmap = shiftedColorMap(cm.seismic, midpoint=midpoint, name='shifted')
# cs = ax.contourf(interv,interv2,G,300,cmap = shifted_cmap)#cmap=cm.seismic,vmax = np.max(G), vmin = np.min(G))
# ax.contour(interv,interv2,G,[0],colors='k')
# if plotlines:
#     plt.vlines(np.arange(0, ambitus, 1.0/subd), interv2[0], interv2[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
#     plt.hlines(np.arange(-12, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
# plt.xlabel('Premier intervalle (en demi-tons)')
# plt.ylabel('Intervalle ajouté au-dessus (en demi-tons)')
# ax.set_ylim(-12,12)
# ax.set_xlim(0,12)
# ax.yaxis.set_ticks(np.arange(-12,13,2))
# ax.xaxis.set_ticks_position('both')
# ax.tick_params(labelbottom=True,labeltop=True)
# plt.title('Gain d\'harmonicité par ajout d\'une 3e note'+'\nK : {}, σ : {} et decr : {}'.format(K,σ,decr))
# # cbar = fig.colorbar(cs)
# plt.show()

#######################
# Carte de descripteur dynamique

# param = 'diffConcordance'
# f0 = 261.0
# ambitus = 12
# plotlines = True
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
# N = len(interv)
# interv2 = np.concatenate([np.arange(-ambitus,0,ε),interv])
# # M = np.zeros((2*N - 1, N))
# #
# # for i in range(N):
# #     for j in range(2*N-1):
# #         freqs1, freqs2 = [f0,f0*2**(interv[i]/12.0)], [f0,f0*2**(interv[i]/12.0), f0*2**((interv[i]+interv2[j])/12.0)]
# #         if param == 'diffConcordance':
# #             M[j,i] = DiffConcordance(freqs1,freqs2)
# #         if param == 'diffRoughness':
# #             M[j,i] = DiffRoughness(freqs1,freqs2)
# #         if param == 'harmonicChange':
# #             M[j,i] = HarmonicChange(freqs1,freqs2)
# #
# # np.save('M_'+ param,M)
#
# with open('M_'+ param +'.npy','rb') as f:
#     M = np.load(f)
#
#
# fig, ax = plt.subplots(1,figsize=(5, 7.5))
# # cs = ax.contourf(interv,interv2,np.log(1 + 5*(M - M.min())/(M.max()- M.min())),256,cmap = cm.jet)
# cs = ax.contourf(interv,interv2,M,256,cmap = cm.jet)
# if plotlines:
#     plt.vlines(np.arange(0, ambitus, 1.0/subd), interv2[0], interv2[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
#     plt.hlines(np.arange(-12, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
# plt.xlabel('Premier intervalle (en demi-tons)')
# plt.ylabel('Intervalle ajouté au-dessus (en demi-tons)')
# ax.set_ylim(-12,12)
# ax.set_xlim(0,12)
# ax.yaxis.set_ticks(np.arange(-12,13,2))
# ax.xaxis.set_ticks_position('both')
# ax.tick_params(labelbottom=True,labeltop=True)
# plt.title('Concordance différentielle : ajout d\'une 3e note'+'\nK : {}, σ : {} et decr : {}'.format(K,σ,decr))
# cbar = fig.colorbar(cs, format='%.1f')
# plt.show()


#######################

# Carte des accords à 3 notes

# param = 'conc3'
# f0 = 261
# ambitus = 12
# plotlines = True
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
# C = Carte(param, ambitus = ambitus,f0 = f0)
# # np.save('Harmonicity_K{}_15'.format(K),C)
# # with open('Harmonicity_K{}.npy'.format(K), 'rb') as f:
# #     C = np.transpose(np.load(f))
# fig, ax = plt.subplots(1,figsize=(9, 7))
# cs = ax.contourf(interv,interv,C,300,cmap=cm.jet)
# if plotlines:
#     plt.vlines(np.arange(0, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
#     plt.hlines(np.arange(0, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.ylabel('Intervalle haut (en demi-tons)')
# # plt.title('Concordance d\'ordre 3 des triades'+'\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# plt.title(param[0].upper() + param[1:] + ' des triades'+'\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# cbar = fig.colorbar(cs)
# plt.show()

#######################
# Carte 3D des accords à 3 notes

# param = 'harmonicity'
# ambitus = 12
# plotlines = False
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
# # C = Carte(param, ambitus = ambitus,f0 = 523)
#
# with open('Harmonicity_K{}.npy'.format(K), 'rb') as f:
#     C = np.load(f)
#
# fig = plt.figure(figsize=(11, 7))
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(interv, interv)
# surf = ax.plot_surface(X,Y,np.log(C-50),rcount = 200, ccount = 200, cmap=cm.jet)
# # surf = ax.contour(X,Y,np.log(1+C),300, cmap=cm.jet)
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.ylabel('Intervalle haut (en demi-tons)')
# ax.zaxis.set_visible(False)
# # plt.title(param[0].upper() + param[1:] + ' des triades'+'\nK : {}, σ : {}, decr : {}'.format(K,σ,decr))
# plt.show()

#######################
# Cartes avec variation paramètres
#
# param = 'tension'
# ambitus = 12
# plotlines = False
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
# C1 = Carte(param, ambitus = ambitus, K = 1)
# C2 = Carte(param, ambitus = ambitus, K = 2)
# C3 = Carte(param, ambitus = ambitus, K = 5)
#
# plt.figure(figsize=(10, 7.5))
#
# ax = plt.subplot(1,3,1)
# cs = ax.contourf(interv,interv,C1,300,cmap=cm.jet)
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.ylabel('Intervalle haut (en demi-tons)')
# plt.title('K : 1')
#
# ax = plt.subplot(1,3,2)
# cs = ax.contourf(interv,interv,C2,300,cmap=cm.jet)
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.title(param[0].upper() + param[1:] + ' des triades, decr : {}'.format(decr)+'\nK : 2')
#
# ax = plt.subplot(1,3,3)
# cs = ax.contourf(interv,interv,C3,300,cmap=cm.jet)
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.title('K : 5')
#
# plt.show()

#######################
# Carte blanche, position des Accords

# ambitus = 6
# plotlines = True
# subd = 2
# interv = np.arange(0,ambitus+ε,ε)
#
# accords = [[4,3],[3,5],[5,4],[3,4],[4,5],[5,3]]
#
#
# fig, ax = plt.subplots(1,figsize=(7, 7))
# if plotlines:
#     plt.vlines(np.arange(0, ambitus+ε , 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
#     plt.hlines(np.arange(0, ambitus+ε, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
# # plt.plot([0,6],[6,0],[0,6],[0,6],alpha=0.7, linestyle=':', linewidth = 1.0, color = 'k')
# plt.plot([0,4],[6,4],[0,4],[0,4],color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0)
# # plt.plot([4,6],[4,0], color = 'k',alpha=0.7, linestyle='--', linewidth = 1.0)
# plt.plot([0,0],[0,6], color = 'k',alpha=0.7, linestyle='--', linewidth = 1.0)
#
# ax.fill_between([0,4],[0,4],[6,4],facecolor='orange', interpolate=True, alpha = 0.8)
#
# def Renv(x,y, color = 'k'):
#     z = 12-x-y
#     plt.plot([x,y,z],[y,z,x], 'o',color = color)
#     plt.plot([x,y],[y,z],[y,z],[z,x],[z,x],[x,y],linestyle='-', linewidth = 1.0, color = color)
#
# def Prime(x,y,color = 'k'):
#     Renv(x,y,color = color)
#     Renv(y,x,color = color)
#
# def Discret(N=3,M=12,classe = 'prime'):
#     liste_acc = Interval_Reduction(Enumerate(N,M-1),M)
#     liste_acc_red = [[((a+[M])[i+1] - (a+[M])[i])*12./M for i in range(len(a))] for a in liste_acc]
#     print(liste_acc_red)
#     for acc in liste_acc_red:
#         ax.scatter(acc[0],acc[1], color = 'k',alpha = 1.)
#
# Discret(M = 2*12)
# # Prime(2,5,'g')
# # Prime(3,4,'b')
# # Prime(4,4,'r')
# # Prime(1,1,'k')
# # Prime(1,3,'brown')
#
# # plt.plot([0,11],[1,1],[8,8],[0,1],linestyle='-', linewidth = 1.0, color = 'brown')
# # plt.plot([8],[1],'o', color = 'brown')
# # plt.plot([0,9],[3,3],[4,4],[3,0],linestyle='-', linewidth = 1.0, color = 'b')
# # plt.plot([4],[3],'o', color = 'b')
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.ylabel('Intervalle haut (en demi-tons)')
# plt.title('Espace des triades')
# ax.set_ylim(0,ambitus)
# ax.set_xlim(0,ambitus)
# plt.show()



#######################
# Plusieurs courbes : variation K

# ambitus = 12
# f0 = 523.0
# C1 = Courbe('roughness', ambitus = ambitus, K = 1, f0 = f0)
# C2 = Courbe('roughness', ambitus = ambitus, K = 3, f0 = f0)
# C3 = Courbe('roughness', ambitus = ambitus, K = 5, f0 = f0)
# C4 = Courbe('roughness', ambitus = ambitus, K = 7, f0 = f0)
# C5 = Courbe('roughness', ambitus = ambitus, K = 9, f0 = f0)
#
# interv = np.arange(0,ambitus+ε,ε)
# # print('Variation : {}'.format(np.std(C)/np.mean(C)))
#
# f, ax = plt.subplots(1,figsize=(11, 5))
# plt.plot(interv,C1,interv,C2,interv,C3,interv,C4,interv,C5,label = [])
# plt.vlines(range(0, ambitus+1), 0, max(C5), alpha=0.4, linestyle='--')
# plt.vlines(7, 0, max(C5), color = 'r', alpha=0.9, linestyle='--')
# plt.xlabel('Intervalle en demi-tons')
# plt.ylabel('Rugosité')
# plt.title('Courbes de rugosité' + '\nNote de référence : C5, ' +'decr : {}'.format(decr))
# # ax.set_ylim(bottom=0)
# # ax.set_yscale('log')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ('1','3','5','7','9'), title = 'Nombre de partiels', loc='upper right',frameon=True, framealpha=0.75)
#
# plt.show()

#######################
# Plusieurs courbes : variation descripteur

# ambitus = 12
# C1 = Courbe('roughness', ambitus = ambitus, f0 = 130)
# C2 = Courbe('roughness', ambitus = ambitus, f0 = 261)
# C3 = Courbe('roughness', ambitus = ambitus, f0 = 523)
#
# interv = np.arange(0,ambitus+ε,ε)
# # print('Variation : {}'.format(np.std(C)/np.mean(C)))
#
# f, ax = plt.subplots(1,figsize=(11, 5))
# plt.plot(interv,C1,interv,C2,interv,C3,label = [])
# plt.vlines(range(0, ambitus+1), 0, max(C1), alpha=0.4, linestyle='--')
# plt.vlines(7, 0, max(C1), color = 'r', alpha=0.9, linestyle='--')
# plt.xlabel('Intervalle en demi-tons')
# plt.ylabel('Rugosité')
# plt.title('Courbes de rugosité' + '\nK = {}, decr : {}'.format(K,decr))
# # ax.set_ylim(bottom=0)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ('C3','C4','C5'), title = 'Note inférieure', loc='upper right',frameon=True, framealpha=0.75)
#
# plt.show()

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

#######################
# Carte blanche, position des classes d'accords

# ambitus = 6
# plotlines = True
# subd = 1
# interv = np.arange(0,ambitus+ε,ε)
#
#
#
# fig, ax = plt.subplots(1,figsize=(7, 7))
# if plotlines:
#     plt.vlines(np.arange(0, ambitus , 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
#     plt.hlines(np.arange(0, ambitus, 1.0/subd), 0, interv[-1], alpha=0.4, linestyle='--', linewidth = 1.0)
# plt.plot([0,6],[0,6],[0,6],[6,0], color = 'k',alpha=0.7, linestyle=':', linewidth = 1.0) ###
# plt.plot([0,4],[6,4], color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0)
# plt.plot([4,6],[4,0], color = 'k',alpha=0.7, linestyle='--', linewidth = 1.0)
# ax.fill_between([0,4,6],[0,0,0],[6,4,0],facecolor='c', interpolate=True, alpha = 0.8)
# # plt.plot([0,2],[6,2],[0,2],[3,2],[3,4],[6,4],[0,4],[6,4], color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0)
# # plt.plot([2,3],[2,0],[2,6],[2,0],[4,6],[4,0],[4,6],[4,3], color = 'k',alpha=0.7, linestyle='--', linewidth = 1.0)
# # plt.plot([0,2],[3,2],[0,4],[6,4],[0,2],[0,2],[3,4],[3,4],[0,3],[6,3], color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0)
#
# # alpha = 0.8
# # ax.fill_between([0,2,3],[0,0,0],[3,2,0],facecolor='c', interpolate=True, alpha = alpha)
# # ax.fill_between([0,4,6],[6,2,0],[6,4,0],facecolor='limegreen', interpolate=True, alpha = alpha)
# # ax.fill_between([0,2,6],[6,2,0],[6,4,0],facecolor='limegreen', interpolate=True, alpha = alpha)
# # ax.fill_between([3,4,6],[6,4,3],[6,6,6],facecolor='y', interpolate=True, alpha = alpha)
# # ax.fill_between([0,2],[0,2],[3,2],facecolor='lightcoral', interpolate=True, alpha = alpha)
# # ax.fill_between([0,3,4],[6,3,4],[6,4.5,4],facecolor='lightcoral', interpolate=True, alpha = alpha)
#
#
# # ax.fill_between([0,2,4],[6,2,4],[6,5,4],facecolor='chocolate', interpolate=True, alpha = alpha) ###
# # plt.plot([0,2],[6,2],[0,4],[6,4], color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0) ###
# # plt.plot([2,4],[2,4], color = 'k',alpha=0.7, linestyle='-', linewidth = 1.0) ###
# # plt.plot([2,6],[2,0],[4,6],[4,0], color = 'k',alpha=0.7, linestyle='--', linewidth = 1.0)
#
# def Renv(x,y,z, color = 'k'):
#     plt.plot([x,y,z],[y,z,x], 'o',color = color)
#     plt.plot([x,y],[y,z],[y,z],[z,x],[z,x],[x,y],linestyle='-', linewidth = 1.0, color = color)
#
# def Prime(x,y,z,color = 'k'):
#     Renv(x,y,z,color = color)
#     Renv(y,x,z,color = color)
#
# Renv(3,3,6,'k')
# Renv(5,2,5,'b')
# Renv(3,1,8,'r')
# Renv(4,3,5,'g')
# Renv(2,2,8,'g')
#
# # Renv(2,2,2,'r')
# # Renv(4,4,4,'r')
# # Renv(1,1,4,'b')
# # Renv(5,5,2,'b')
# # Renv(3,3,6,'m')
#
#
#
# plt.xlabel('Classe d\'intervalles basse (en demi-tons)')
# plt.ylabel('Classe d\'intervalles haute (en demi-tons)')
# plt.title('Espace des formes réduites de triades')
# ax.set_ylim(0,ambitus)
# ax.set_xlim(0,ambitus)
# plt.show()

##########################
# Courbes structure harmonicité

# x = np.arange(0,12,0.1)
# y1 = np.zeros(len(x))
# y2 = np.zeros(len(x))
# y3 = np.zeros(len(x))
#
# for i in range(len(x)):
#     y1[i] = - 12.*np.log2(2-2**(x[i]/12.))
#     y2[i] = 12 - 12.*np.log2(3-2**(x[i]/12.))
#     y3[i] = - 12.*np.log2(3-2*2**(x[i]/12.))
#
# f, ax = plt.subplots(1,figsize=(7, 7))
# ax.set_ylim(0,12)
# ax.set_xlim(0,12)
# plt.plot(x,y1,x,y2,x,y3, label = [])
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ('groupe 1','groupe 2','groupe 3'),frameon=True, framealpha=0.75)
# plt.xlabel('Intervalle bas (en demi-tons)')
# plt.ylabel('Intervalle haut (en demi-tons)')
# plt.show()

##########################
