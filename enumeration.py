

'''
Script qui énumère tous les accords à N notes partant de do médium avec un ambitus maximum M
'''

from music21 import *
import numpy as np
import matplotlib.pyplot as plt

N = 3 # Nombre de notes
M = 11 # Ambitius (en demi-tons)

''' Fonction qui fait donne la liste de tous les accords, sous forme de PitchClass  '''
def Enumerate(N,M):
    Liste_Accords = []
    def Add_notes(NotesPrec, N, M):
        # Liste complète : on a un accord, qu'on ajoute à Liste_Accords
        if len(NotesPrec)==N:
            Liste_Accords.append(NotesPrec)
        # Encore des notes à ajouter
        else:
            for i in range(NotesPrec[-1]+1, M-N+2+len(NotesPrec)):
                Add_notes(NotesPrec + [i], N, M)
    Add_notes([0],N,M)
    return Liste_Accords

def Normal_Reduction(ListeAccords):
    ListeAccordsInt = [[(a+[12])[i+1] - (a+[12])[i] for i in range(len(a))] for a in ListeAccords]
    ListeNormalFormInt = []
    def normal_form(a):
        def cyclic_perm(a):
            n = len(a)
            b = [[a[i - j] for i in range(n)] for j in range(n)]
            return b
        l = cyclic_perm(a)
        for a in l: a.reverse()
        l.sort(reverse = True)
        l[0].reverse()
        return l[0]

    for a in ListeAccordsInt:
        a_norm = normal_form(a)
        if a_norm not in ListeNormalFormInt:
            ListeNormalFormInt.append(a_norm)
    ListeNormalForm = [[sum(([0]+a)[:i]) for i in range (1,len(a)+1)] for a in ListeNormalFormInt]
    return ListeNormalForm


def Make_score(ListeAccords, permute = False):
    stream1 = stream.Stream()
    if not permute:
        for i, accord in enumerate(ListeAccords):
            stream1.append(chord.Chord(accord))
    else:
        ListeAccords_perm = np.random.permutation(ListeAccords).tolist()
        for i, accord in enumerate(ListeAccords_perm):
            stream1.append(chord.Chord(accord))
    stream1.show()

# ListeAccords = [[4,3,5],[3,]]
# ListeNormalForm = Normal_Reduction(ListeAccords)

ListeAccords = Enumerate(N,M)
ListeNormalForm = Normal_Reduction(ListeAccords)
Make_score(ListeNormalForm, True)
