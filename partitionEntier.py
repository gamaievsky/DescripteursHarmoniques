import numpy as np
import matplotlib.pyplot as plt

dic = {}

def Colonne(N,M):
    if (N,M) in dic:
        P = dic[(N,M)]
    elif N>M : P = 0
    elif (N==M or N==1) : P = 1
    else:
        P = Colonne(N-1, M-1) + Colonne(N, M-N)
        dic[(N,M)] = P
    return P

def Partition(M):
    P = 0
    for N in range(1,M+1):
        P += Colonne(N,M)
    return P

############### Écrire les nombres d'identités #####################
part = []
print('Nombre d\'identités en fonction de la subdivision :')
for M in range(1,62+1):
    P = Partition(M)
    part.append(P)
#     print('N = {} : {} identités'.format(M,P))
# print('\n--------------------------\n')

plt.figure()
plt.plot(range(1,62+1), part)
plt.yscale('log')
plt.xlabel('Échelle : Subdivisions de l\'octave')
plt.ylabel('Nombre d\'identités')
plt.show()
