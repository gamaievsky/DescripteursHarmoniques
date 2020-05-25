

'''
Script qui énumère tous les accords à N notes partant de do médium avec un ambitus maximum M
'''

from music21 import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from music21.converter.subConverters import ConverterMusicXML
from PIL import Image
import os
from p5 import *
import itertools

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
    def Normal_form(a):
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
        a_norm = Normal_form(a)
        if a_norm not in ListeNormalFormInt:
            ListeNormalFormInt.append(a_norm)
    ListeNormalForm = [[sum(([0]+a)[:i]) for i in range (1,len(a)+1)] for a in ListeNormalFormInt]
    return ListeNormalForm

def Interval_Reduction(ListeAccords):
    ListeAccordsIntMod = []
    ListeAccordsMod = [[(a+[12])[i+1] - (a+[12])[i] for i in range(len(a))] for a in ListeAccords]
    for a in ListeAccordsMod:
        a.sort()
        if a not in ListeAccordsIntMod:
            ListeAccordsIntMod.append(a)
    ListeAccordsInt = [[sum(([0]+a)[:i]) for i in range (1,len(a)+1)] for a in ListeAccordsIntMod]
    return ListeAccordsInt




def Make_score(ListeAccords, permute = False, print = 'intervalVector',indStart = 2):
    stream1 = stream.Stream()
    ListeAccordsMod = [[(a+[12])[i+1] - (a+[12])[i] for i in range(len(a))] for a in ListeAccords]
    ind = indStart
    if not permute:
        for i, accord in enumerate(ListeAccords):
            c = chord.Chord(accord)
            l = sorted(ListeAccordsMod[i])
            c.quarterLength = 4.0
            c.lyric = '{}: '.format(ind) + str(l)
            stream1.append(c)
            ind += 1
    else:
        ListeAccords_perm = np.random.permutation(ListeAccords).tolist()
        for i, accord in enumerate(ListeAccords_perm):
            c = chord.Chord(accord)
            c.duration = duration.Duration(4)
            stream1.append(chord.Chord(c))
    # for acc in stream1:
    #     if acc.isChord:
    #         acc.lyric('lyric')
    stream1.show()

# ListeAccords = [[4,3,5],[3,]]
# ListeNormalForm = Normal_Reduction(ListeAccords)
#
# ListeAccords = Enumerate(N,M)
# # print(ListeAccords)
# # ListeNormalForm = Normal_Reduction(ListeAccords)
# ListeAccordsInt = Interval_Reduction(ListeAccords)
# Make_score(ListeAccordsInt, False, indStart = 77)

# ListeIntervalVector = []
# for N in range(1,13):
#     ListeAccords = Enumerate(N,M)
#     ListeAccordsInt = Interval_Reduction(ListeAccords)
#     ListeIntervalVector += ListeAccordsInt
# Make_score(ListeIntervalVector, False, indStart = 1)





################# Enregistrer toutes les images #########################
liste_imWidth = [260,330,370,400,450,500,540,600,640,680,730,780]
liste_imWidth_start = [0,130,130,130,130,130,130,130,130,130,130,130]
dic_imWidth = {i:(liste_imWidth_start[i-1],liste_imWidth[i-1]) for i in range(1,13)}
conv_musicxml = ConverterMusicXML()

def AccordsPremierNiveau():
    ind = 1
    for N in range(1,13):
        left, right = dic_imWidth[N]
        ListeAccords = Enumerate(N,M)
        ListeAccordsInt = Interval_Reduction(ListeAccords)
        for a in ListeAccordsInt:
            stream1 = stream.Stream()
            a_mod = [(a+[12])[i+1] - (a+[12])[i] for i in range(len(a))]
            c = chord.Chord(a)
            c.quarterLength = 4.0
            c.lyric = str('{}: '.format(ind) + str(a_mod))
            stream1.append(c)

            title = 'acc{}'.format(ind)
            filepath = '/Users/manuel/Dropbox (TMG)/Thèse/Estrada/'
            out_filepath = conv_musicxml.write(stream1, 'musicxml', fp=filepath + title + '.XML', subformats=['png'])

            im = Image.open(out_filepath)
            width, height = im.size
            region = im.crop((left, 0, right, height))
            region.save(out_filepath[:-6] + '.png')

            os.remove(filepath + title + '.XML')
            os.remove(filepath + title + '-1.png')

            ind += 1

def AccordsDeuxiemeNiveau():

    def cyclic_perm(a):
        n = len(a)
        b = [[a[i - j] for i in range(n)] for j in range(n)]
        return b

    def Normal_form(a):
        l = cyclic_perm(a)
        for a in l: a.reverse()
        l.sort(reverse = True)
        l[0].reverse()
        return l[0]

    def Normal_Augmentation(a):
        a_mod = [(a+[12])[i+1] - (a+[12])[i] for i in range(len(a))]
        liste_perm = list(itertools.permutations(a_mod))
        ListeAccordsNorm = []
        for acc_perm in liste_perm:
            acc_perm_norm = Normal_form(acc_perm)
            if acc_perm_norm not in ListeAccordsNorm:
                ListeAccordsNorm.append(acc_perm_norm)
        return ListeAccordsNorm

    ind = 1
    for N in range(1,13):
        left, right = dic_imWidth[N]
        ListeAccords = Enumerate(N,M)
        ListeAccordsInt = Interval_Reduction(ListeAccords)
        for a in ListeAccordsInt:
            liste_a_augm = Normal_Augmentation(a)
            perm = 1
            for a_perm in liste_a_augm:
                a_perm_nonMod = [sum(([0]+a_perm)[:i]) for i in range (1,len(a_perm)+1)]
                stream1 = stream.Stream()
                c = chord.Chord(a_perm_nonMod)
                c.quarterLength = 4.0
                c.lyric = str(a_perm)
                stream1.append(c)

                title = 'acc{}-{}'.format(ind,perm)
                filepath = '/Users/manuel/Dropbox (TMG)/Thèse/Estrada/'
                out_filepath = conv_musicxml.write(stream1, 'musicxml', fp=filepath + title + '.XML', subformats=['png'])
                im = Image.open(out_filepath)
                width, height = im.size
                region = im.crop((left, 0, right, height))
                region.save(out_filepath[:-6] + '.png')

                os.remove(filepath + title + '.XML')
                os.remove(filepath + title + '-1.png')

                perm += 1


            ind += 1





#######################

AccordsDeuxiemeNiveau()





# Make_score(ListeIntervalVector, False, indStart = 1)


# ListeIntervalVector = [[(a+[12])[i+1] - (a+[12])[i] for i in range(len(a))] for a in ListeIntervalVector]
#
# with open('liste_interval_vectors.pkl', 'wb') as f:
#     pickle.dump(ListeIntervalVector, f)
