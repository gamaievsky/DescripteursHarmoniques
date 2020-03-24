

'''
Script qui énumère tous les accords à N notes partant de do médium avec un ambitus maximum M
'''

from music21 import *
import numpy as np
import matplotlib.pyplot as plt

N = 5 # Nombre de Nombre de notes
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

ListeAccords = Enumerate(N,M)
print(len(ListeAccords))
Make_score(ListeAccords, True)
