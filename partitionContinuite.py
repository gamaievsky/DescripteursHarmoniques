
'''
Etant donnée une liste d'accords, les trie par ordre de concordance, consonance, tension et concordance totale,
en affichant en dessous les valeurs. Prend en entrée dans le fichier paramètre deux listes de même taille : partiels,
qui contient l'emplacement des partiels (éventuellement inharmoniques),  et amplitudes, avec leurs amplitudes
respectives
'''

#from mimetypes import init

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from operator import itemgetter, attrgetter
from music21 import *
import os
#from music21 import note, stream, corpus, tree, chord, pitch, converter

import parametres
import tunings
listeSpectresAccords = []


class ListeAccords:
    '''
    Prend en attribut un objet Stream, les parametres d'instrument et d'accord:

    instrument = [liste des partiels, liste des amplitudes, liste des largeurs spectrales].
    temperament = [Nom de la note de reference e partir de laquelle va etre fait l'accord, liste des deviations en cents des 11 notes
                    restantes par rapport au temperament tempere]

    L'attribut 'grandeursHarmoniques' va stoker sous forme de liste les informations de concordance et de coherence de chaque accord,
    ainsi que son nombre de notes
    L'attribut normalisation va servir a normaliser les concordances des unissons de n notes
    '''


    def __init__(self, stream, instr = parametres.timbre_def, partition = ''):

        self.stream = stream
        self.tree = tree.fromStream.asTimespans(stream, flatten=True,classList=(note.Note, chord.Chord))
        if parametres.mode=='manual':
            self.partiels = parametres.partiels
            self.amplitudes = parametres.amplitudes
            self.sig = parametres.sig
        else:
            self.partiels = []
            self.amplitudes = []
            self.K = instr[0]
            self.decr = instr[1]
            self.sig = instr[2]
            if not parametres.shepard:
                for i in range(self.K):
                    self.partiels.append(i+1)
                    self.amplitudes.append(1/(i+1)**self.decr)


        self.temperament = tunings.Equal
                            #[0,0,0,0,0,0,0,0,0,0,0,0]#Tempere#
                            #[0,-10,+4,-6,+8,-2,-12,+2,-8,+6,-4,+10]#Pythagore
                            #[0,+17,-7,+10,-14,+3,+20,-3,+14,-10,+7,-17]#Mesotonique 1/4
                            #[]#Mesotonique 1/6
                            #[0,-29,+4,+16,-14,-2,-31,+2,-27,-16,-4,-12]#Juste Majeur
                            #[0,12,+4,+16,-13,-2,+32,+2,+14,-17,+18,-11]#Juste mineur

        self.noteDeReferencePourLeTunning = parametres.noteDeReferencePourLeTunning
        self.grandeursHarmoniques = []
        self.energyUniss = [2,3,4,5,6,7,8]
        self.partition = partition


    def spectre(self,f0):

        '''Cette methode va etre appelee dans la classe Accord, mais elle est definie ici car le seul attribut d'objet
         qui en est parametre est l'instrument'''
        n = np.arange(0,16,0.001)
        S = np.zeros(np.shape(n))

        if not parametres.shepard:
            for i in range(1, len(self.partiels) + 1):
                S = S + (self.amplitudes[i-1]) * np.exp(-(n - np.log2(self.partiels[i-1] * f0))**2 / (2 * self.sig**2))

        else:
            def repr_classe(f0,fmin,fmax):
                if fmin <= f0 < fmax: return f0
                elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
                elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
            f0 = repr_classe(f0,261.0,522.0)
            p0 = np.log2(261)
            Σ = 3.0
            E = np.exp(-(n - p0)**2 / (2 * Σ**2))

            for k in range(1,self.K+1):
                f = repr_classe(k*f0,261.0,522.0)
                p = np.log2(f)
                for i in range(-8,8):
                    if 0 < p +i < 16:
                        S += (1/k**self.decr) * np.exp(-(n - (p+i))**2 / (2 * self.sig**2))
        return S


    def spectre_pic(self,f0):

        dic_spectre = {}
        if parametres.shepard:
            def repr_classe(f0,fmin,fmax):
                if fmin <= f0 < fmax: return f0
                elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
                elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
            f = repr_classe(f0,261.0,522.0)
            p0 = np.log2(261)
            Σ = 3.0

            # Construction de dic_spectre
            for k in range(1,self.K+1):
                f = repr_classe(k*f0,261.0,522.0)
                p = np.log2(f)
                for i in range(-8,8):
                    if 0 < p + i < 16:
                        fp = f*2**i
                        if fp in dic_spectre:
                            dic_spectre[fp] += (1/k**self.decr)* np.exp(-(np.log2(fp) - p0)**2 / (2 * Σ**2))
                        else : dic_spectre[fp] = (1/k**self.decr)* np.exp(-(np.log2(fp) - p0)**2 / (2 * Σ**2))
        else:
            for k in range(1, self.K+1):
                fk = k*f0
                dic_spectre[fk] = self.amplitudes[k-1]


        return dic_spectre


    def EnergyUniss(self):

        """ Calcule la concordance d'ordre n de l'unisson a n notes, pour n allant de 2 a 8"""

        self.energyUniss[0] = np.sum(self.spectre(100)*self.spectre(100))
        self.energyUniss[1] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/3)
        self.energyUniss[2] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/4)
        self.energyUniss[3] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/5)
        self.energyUniss[4] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/6)
        self.energyUniss[5] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/7)
        self.energyUniss[6] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/8)


    def frequenceAvecTemperament(self,pitch1):

        """Fonction qui prend en entree un pitch pour renvoyer une frequence, en tenant compte du temperament"""

        pitchRef = pitch.Pitch(self.noteDeReferencePourLeTunning)
        pitch1.microtone = self.temperament[(pitch1.pitchClass - pitchRef.pitchClass)%12] - 100*((pitch1.pitchClass - pitchRef.pitchClass)%12)

        return (pitch1.frequency)


    def HarmonicDescriptors (self):

        ''' Transforme chaque verticalite en objet Accord, calcule la concordance, la coherence et les concordances multiples, et stocke les resultats
        sous forme de liste d'Accords"
        '''

        self.EnergyUniss()
        Prec = 0

        for verticality in self.tree.iterateVerticalities():
            v = Accord(verticality, self.partiels, self.amplitudes, self.K, self.sig, self.decr, self.energyUniss, self.temperament,self.noteDeReferencePourLeTunning)
            if verticality.bassTimespan!=None :
                v.identifiant = verticality.bassTimespan.element.id

            v.ListeHauteursAvecMultiplicite()
            v.NombreDeNotes()

            if v.nombreDeNotes>=1:
                v.SpectreAccord()
                listeSpectresAccords.append(v.spectreAccord)
                v.Context()
                # v.Roughness()
                # v.SpectreConcordance()
                # v.Concordance()
                # v.Harmonicity()
                # v.CrossConcordance(Prec)
                v.HarmonicChange(Prec)
                # v.HarmonicNovelty(Prec)
                v.DiffConcordance(Prec)
                v.DiffConcordanceContext(Prec)
                # v.DiffRoughness(Prec)

                if v.nombreDeNotes>=3:
                    # v.ConcordanceOrdre3()
                    v.SpectreConcordanceTot()
                    v.ConcordanceTotale()
                    # v.Tension()
                    # v.CrossConcordanceTot(Prec)

            Prec = v
            self.grandeursHarmoniques.append(v)


    def Liste(self, axe = 'concordance'):
        liste = []
        for accord in self.grandeursHarmoniques:
            if isinstance(getattr(accord, axe),str): pass
            else: liste.append(getattr(accord, axe))
        return liste

    def getAnnotatedStream(self, space = ['concordance'], write_name = True):
        r=0
        i = 0
        liste = [self.Liste(descr) for descr in space]
        for j in range(len(liste)):
            m = max(liste[j])
            if m != 0:
                pass
                liste[j] = [(100/m)*val for val in liste[j]]
        print(liste)
        for gH in self.grandeursHarmoniques:
            if gH.verticality.bassTimespan != None :
                element = gH.verticality.bassTimespan.element
                if element.isNote or element.isChord:
                    dataString = ""

                for d, descr in enumerate(space):
                    if dataString != '': dataString + " "
                    #Descripteurs différentiels
                    if descr in ['harmonicChange', 'diffConcordanceContext']:
                        if type(getattr(gH,descr)) != str:
                            # dataString = dataString + "-" + str(round(liste[d][i],8))
                            dataString = dataString + "-" + str(int(liste[d][i-1]))

                    #Descripteurs statiques
                    else:
                        # dataString = dataString + str(round(liste[d][i],5))
                        dataString = dataString + str(int(liste[d][i]))

                #Rajout du nom du descripteur
                if r == 0:
                    if write_name:
                        if parametres.shepard: shep = 'Shepard, '
                        else: shep = 'no Shepard, '
                        dataString = dataString  + "\n" + space[0][0].upper() + space[0][1:] +'\n'+ shep + 'K : {}, σ : {}, decr : {}'.format(self.K,self.sig,self.decr)
                        r=1

                #Assignement
                element.lyric = dataString
                # element.lyric = element.forteClass
                # element.lyric = element.intervalVector
            i+=1
        return tree.toStream.partwise(self.tree, self.stream)


    def Classement(self, descr, reverse = True):
        s = stream.Measure()
        ts1 = meter.TimeSignature('C')
        s.insert(0, ts1)
        s.insert(0, clef.TrebleClef())

        self.stream.lyrics = {}
        self.getAnnotatedStream([descr], write_name = False)
        self.grandeursHarmoniques.sort(key=attrgetter(descr), reverse = reverse)
        for gH in self.grandeursHarmoniques:
            element = gH.verticality.bassTimespan.element
            s.insert(-1, element)
        if parametres.shepard: shep = 'Shepard, '
        else: shep = 'no Shepard, '
        s[0].addLyric(descr[0].upper() + descr[1:] + ' triée' + '\n'+ shep + 'K : {}, σ : {}, decr : {}'.format(self.K,self.sig,self.decr))
        s.show()
        del s[0].lyrics[1]


    '''Fonction qui regroupe dans un tableau de dimension d*T ou d*(T-1) les valeurs des d descripteurs sur une séquence de T accords
    '''
    def Points(self,space):
        d = len(space)
        L = []
        for descr in space:
            L.append(self.Liste(descr))

        # Si à la fois descripteurs statiques et dynamiques dans space, alors on réduit la longueur des listes de descripteurs statiques en considérant l'évolution du descripteur
        T = min(map(len,L))
        for i in range(d):
            if len(L[i])>T:
                for t in range(len(L[i])-1):
                    L[i][t] = L[i][t+1] - L[i][t]
                L[i].pop(-1)

        Points = np.asarray(L)
        return Points


    def TemporalRepresentation(self, space):

        # Construction de l'axe temporel : détection des ONSETS
        times = [gH.verticality.offset for gH in self.grandeursHarmoniques]
        gH = self.grandeursHarmoniques[len(self.grandeursHarmoniques) - 1]
        times.append(gH.verticality.bassTimespan.endTime)
        n_frames = len(times)-1

        dim = len(space)
        plt.figure(figsize=(13, 7))
        plt.title('K : {}, σ : {}, decr : {}'.format(self.K,self.sig,self.decr))

        # plt.title(title)
        s = int(parametres.aff_score and (len(self.partition) != 0))

        if s:
            img=mpimg.imread(self.partition)
            score = plt.subplot(dim+s,1,1)
            plt.axis('off')
            score.imshow(img)
            plt.title('K : {}, σ : {}, decr : {}'.format(self.K,self.sig,self.decr))


        for k, descr in enumerate(space):
            l = self.Liste(descr)
            if "" in l: l.remove("")
            plt.subplot(dim+s, 1, k+1+s)
            plt.vlines(times, 0, max(l), color='k', alpha=0.9, linestyle='--')
            if not all(x>=0 for x in l):
                plt.hlines(0,times[0], times[n_frames], alpha=0.5, linestyle = ':')

            # Descripteurs Statiques
            if len(l) == n_frames:
                plt.hlines(l, times[:(n_frames)], times[1:n_frames+1], color=['b','r','g','c','m','y','b','r','g'][k], label=descr[0].upper() + descr[1:])

            # Descripteurs dynamiques
            elif len(l) == (n_frames-1):
                plt.plot(times[1:n_frames], l,['b','r','g','c','m','y','b','r','g'][k]+'o', label=descr[0].upper() + descr[1:])
                plt.hlines(l, [t-0.25 for t in times[1:n_frames]], [t+0.25 for t in times[1:n_frames]], color=['b','r','g','c','m','y','b','r','g'][k], alpha=0.9, linestyle=':')

            plt.legend(frameon=True, framealpha=0.75)
            # ax.set_ylim(bottom=0)
        plt.legend(frameon=True, framealpha=0.75)
        plt.tight_layout()
        plt.show()


    def AbstractRepresentation(self, space =  ['concordance','Roughness','tension']):
        if len(space)==2 :
            color = parametres.color_abstr
            l1 = self.Liste(space[0])
            l2 = self.Liste(space[1])


            #Si un descripteur statique et un descripteur dynamique
            if len(l1)<len(l2) : l2.pop(0)
            elif len(l1)>len(l2) : l1.pop(0)

            fig, ax = plt.subplots()
            if parametres.link: plt.plot(l1, l2, color+'--')
            plt.plot(l1, l2, color+'o')
            for i in range(len(l1)):
                ax.annotate(' {}'.format(i+1), (l1[i], l2[i]))
            plt.xlabel(space[0][0].upper() + space[0][1:])
            plt.ylabel(space[1][0].upper() + space[1][1:])
            plt.title(title)
            plt.show()
        else:
            color = parametres.color_abstr
            l1 = self.Liste(space[0])
            l2 = self.Liste(space[1])
            l3 = self.Liste(space[2])
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            if parametres.link: plt.plot(l1, l2, l3, color+'--')
            for i in range(len(l1)):
                ax.scatter(l1[i], l2[i], l3[i], c=color, marker='o')
                ax.text(l1[i], l2[i], l3[i], i+1, color='red')
            ax.set_xlabel(space[0])
            ax.set_ylabel(space[1])
            ax.set_zlabel(space[2])
            ax.set_title('title')
            plt.show()


    def Affichage(self, space):
        if parametres.plot_score:
            self.getAnnotatedStream(space)
            self.stream.show()

        if parametres.plot_class:
            self.Classement(space[0])

        if parametres.plot_descr:
            self.TemporalRepresentation(space)
        if parametres.plot_abstr:
            self.AbstractRepresentation(space)




class Accord(ListeAccords):
    '''
    Classe qui traite les verticalites en heritant de l'instrument et de la methode spectre de la classe ListeAccords,
    et ayant comme attributs supplementaires les grandeurs lies a la concordance
    Faiblesse pour l'instant : l'arbre de la classe mere est vide, un attribut 'verticality' vient le remplacer
    '''

    def __init__(self, verticality, partiels, amplitudes,K, sig, decr, energyUniss, temperament, noteDeReferencePourLeTunning):

        self.partiels = partiels
        self.amplitudes = amplitudes
        self.K = K
        self.sig = sig
        self.decr = decr
        self.temperament = temperament
        self.noteDeReferencePourLeTunning = noteDeReferencePourLeTunning
        self.energy = 0
        self.energyUniss = energyUniss
        self.listeHauteursAvecMultiplicite = []
        self.verticality = verticality
        self.concordance = 0
        self.concordanceTotale = 0
        self.concordanceOrdre3 = 0
        self.roughness = 0
        self.tension = 0
        self.identifiant = 0
        self.nombreDeNotes = 0
        self.spectreConcordance = 0
        self.spectreConcordanceTot = 0
        self.crossConcordance = 0
        self.crossConcordanceTot = 0
        self.baryConc= 0
        self.baryConcTot = 0
        self.difBaryConc = 0
        self.difBaryConcTot = 0
        self.spectreAccord = 0
        self.context = 0
        self.energyContext = 0
        self.harmonicChange = 0
        self.diffConcordance = 0
        self.diffConcordanceContext = 0
        self.diffRoughness = 0
        self.spectreHarmonicNovelty = 0
        self.harmonicNovelty = 0
        self.harmonicity = 0






    def __repr__(self):
        """Affichage"""
        return "Concordance: {0} \nConcordance d'ordre 3:  {2} \nConcordance totale: {3}".format(self.concordance,self.concordanceOrdre3,self.concordanceTotale)

    def ListeHauteursAvecMultiplicite(self):

        """ Fonction qui donne la liste des pitches, comptes autant de fois qu'ils sont repetes a differentes voix"""

        #self.listeHauteursAvecMultiplicite = list
        for elt in self.verticality.startTimespans:
            if elt.element.isChord:
                for pitch in elt.element.pitches:
                    if elt.element.duration.quarterLength != 0:
                        self.listeHauteursAvecMultiplicite.append(pitch)

            elif elt.element.duration.quarterLength != 0:
                 self.listeHauteursAvecMultiplicite.append(elt.element.pitch)

        for elt in self.verticality.overlapTimespans:
            if elt.element.isChord:
                for pitch in elt.element.pitches:
                    self.listeHauteursAvecMultiplicite.append(pitch)

            else:
                 self.listeHauteursAvecMultiplicite.append(elt.element.pitch)

    def NombreDeNotes(self):
        if self.listeHauteursAvecMultiplicite != None:
            self.nombreDeNotes = len(self.listeHauteursAvecMultiplicite)

    # def Roughness(self):
    #
    #     for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
    #         for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
    #             if (i<j):
    #                 for k1 in range(1,len(self.partiels) + 1):
    #                     for k2 in range(1,len(self.partiels) + 1):
    #                         freq = [self.partiels[k2-1] * self.frequenceAvecTemperament(pitch2), self.partiels[k1-1] * self.frequenceAvecTemperament(pitch1)]
    #                         freq.sort()
    #                         fmin, fmax = freq[0], freq[1]
    #                         s = 0.44*(np.log(parametres.β2/parametres.β1)/(parametres.β2-parametres.β1))*(fmax-fmin)/(fmin**(0.477))
    #                         diss = np.exp(-parametres.β1*s)-np.exp(-parametres.β2*s)
    #                         if parametres.type_diss=='produit':
    #                             self.roughness = self.roughness + (self.amplitudes[k1-1] * self.amplitudes[k2-1]) * diss
    #                         elif parametres.type_diss == 'minimum':
    #                             self.roughness = self.roughness + min(self.amplitudes[k1-1],self.amplitudes[k2-1]) * diss
    #
    #      if parametres.norm_diss:
    #          if parametres.type_diss=='produit':
    #              self.roughness = self.roughness/self.energy
    #          elif parametres.type_diss == 'minimum':
    #              self.roughness = self.roughness/np.sqrt(self.energyUniss[0])
    #
    #      n = self.nombreDeNotes
    #      self.roughness = self.roughness/(self.nombreDeNotes*(self.nombreDeNotes - 1)/2)

    def Roughness(self):

        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                if (i<j):
                    dic1, dic2 = self.spectre_pic(self.frequenceAvecTemperament(pitch1)), self.spectre_pic(self.frequenceAvecTemperament(pitch2))
                    for f1 in dic1:
                        for f2 in dic2:
                            freq = [f1,f2]
                            freq.sort()
                            fmin, fmax = freq[0], freq[1]
                            s = 0.44*(np.log(parametres.β2/parametres.β1)/(parametres.β2-parametres.β1))*(fmax-fmin)/(fmin**(0.477))
                            diss = np.exp(-parametres.β1*s)-np.exp(-parametres.β2*s)
                            if parametres.type_diss=='produit':
                                self.roughness = self.roughness + (dic1[f1] * dic2[f2]) * diss
                            elif parametres.type_diss == 'minimum':
                                self.roughness = self.roughness + min(dic1[f1],dic2[f2]) * diss

        if parametres.norm_diss:
            if parametres.type_diss=='produit':
                self.roughness = self.roughness/self.energy
            elif parametres.type_diss == 'minimum':
                self.roughness = self.roughness/np.sqrt(self.energyUniss[0])

        n = self.nombreDeNotes
        self.roughness = self.roughness/(self.nombreDeNotes*(self.nombreDeNotes - 1)/2)


    def Tension(self):
        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                for l, pitch3 in enumerate(self.listeHauteursAvecMultiplicite):
                    if (i<j<l):
                        dic1, dic2, dic3 = self.spectre_pic(self.frequenceAvecTemperament(pitch1)), self.spectre_pic(self.frequenceAvecTemperament(pitch2)), self.spectre_pic(self.frequenceAvecTemperament(pitch3))
                        for f1 in dic1:
                            for f2 in dic2:
                                for f3 in dic3:
                                    freq = [f1,f2,f3]
                                    freq_ord = sorted(range(len(freq)), key=lambda k: freq[k])
                                    dic_ord = [[dic1, dic2, dic3][i] for i in freq_ord]
                                    freq.sort()
                                    x = np.log2(freq[1] / freq[0])
                                    y = np.log2( freq[2] / freq[1])

                                    self.tension += (dic_ord[0][freq[0]] * dic_ord[1][freq[1]] * dic_ord[2][freq[2]]) * np.exp(-(12*(abs(y-x))/parametres.δ)**2)

        n = self.nombreDeNotes
        self.concordanceOrdre3 *= (n**3 / (n *(n-1)*(n-2)/6)) / LA.norm(self.spectreAccord,ord=3)**3

    def Harmonicity(self):#cas Shepard
        f0 = 261
        # Division du demi-ton en 8e de demi-tons
        freqOct = [f0*2**(i/(12*8)) for i in range(12*8)]
        for f in freqOct:
            corr = np.sum(np.multiply(self.spectreAccord, self.spectre(f)))
            if self.harmonicity < corr:
                self.harmonicity = corr
        self.harmonicity /= self.energy


    def ConcordanceOrdre3(self):

        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                for k, pitch3 in enumerate(self.listeHauteursAvecMultiplicite):
                    if (i<j<k):
                        self.concordanceOrdre3 =  self.concordanceOrdre3 + np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))*self.spectre(self.frequenceAvecTemperament(pitch3)))

        self.concordanceOrdre3 = self.concordanceOrdre3**(2/3)
        #self.concordanceOrdre3 = np.log2(1 + self.concordanceOrdre3 / (self.energyUniss[1]*(self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)**(2/3)))
        #self.concordanceOrdre3 = np.log2(1 + self.concordanceOrdre3)/(np.log(1 + self.energyUniss[1] * (self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)**(2/3)) / np.log(1 + self.energyUniss[1]))

    def SpectreConcordance(self):
        self.spectreConcordance = np.zeros(16000)
        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                if (i<j):
                    self.spectreConcordance = self.spectreConcordance + self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))
        self.spectreConcordance = self.spectreConcordance / self.energy

    def Concordance(self):

        " Normalisation logarithmique, de maniere a rendre egales les concordances des unissons de n notes"
        self.concordance = np.sum(self.spectreConcordance)
        self.concordance = self.concordance*(self.nombreDeNotes)/((self.nombreDeNotes - 1)/2)

        #self.concordance = np.log2(1 + self.concordance / (self.energyUniss[0]*self.nombreDeNotes*(self.nombreDeNotes - 1)/2))
        #self.concordance = np.log2(1 + self.concordance)/(np.log(1 + self.energyUniss[0]*self.nombreDeNotes*(self.nombreDeNotes - 1)/2) / np.log(1 + self.energyUniss[0]))

    def CrossConcordance(self, Prec):
        if type(Prec) is int:
            self.crossConcordance = ""
        elif parametres.norm_crossConc=='first': # Version normalisée par rapport à la première concordance
            self.crossConcordance = (np.sum(self.spectreConcordance * Prec.spectreConcordance) / np.sum(Prec.spectreConcordance * Prec.spectreConcordance))
        elif parametres.norm_crossConc=='energy + conc':
            self.crossConcordance = np.divide(np.sum(self.spectreConcordance * Prec.spectreConcordance), np.sum(Prec.spectreConcordance)*self.concordance/(Prec.nombreDeNotes*(Prec.nombreDeNotes - 1)/2))
        elif parametres.norm_crossConc=='energy':
            self.crossConcordance = np.sum(self.spectreConcordance * Prec.spectreConcordance)


    def BaryConcordance(self):
        self.baryConc = np.sum(self.spectreConcordance * np.nnènhe(0,16,0.001)) / (self.concordance * (self.nombreDeNotes*(self.nombreDeNotes - 1)/2))


    def DifBaryConc(self,Prec): # Calcule les variations de barycentre
        if type(Prec) is int:
            self.difBaryConc = ""
        else:
            self.difBaryConc = self.baryConc - Prec.baryConc
        #if self.difBaryConc != "": print(round(self.difBaryConc,2))

    def SpectreConcordanceTot(self):
        self.spectreConcordanceTot = np.ones(16000)
        for pitch in self.listeHauteursAvecMultiplicite:
                self.spectreConcordanceTot = self.spectreConcordanceTot * self.spectre(self.frequenceAvecTemperament(pitch))


    def ConcordanceTotale(self):

          S = np.ones(16000)
          for pitch in self.listeHauteursAvecMultiplicite:
                  S = S*self.spectre(self.frequenceAvecTemperament(pitch))

          self.concordanceTotale = np.sum(self.spectreConcordanceTot)**(2/self.nombreDeNotes)


    def CrossConcordanceTot(self, Prec):
        if type(Prec) is int:
            self.crossConcordanceTot = ""

        elif parametres.norm_crossConcTot=='first': # Version normalisée par rapport à la première concordance
            self.crossConcordanceTot = np.divide(np.sum(self.spectreConcordanceTot * Prec.spectreConcordanceTot)**(2/(self.nombreDeNotes + Prec.nombreDeNotes)), np.sum(Prec.spectreConcordanceTot * Prec.spectreConcordanceTot)**(2/(2*Prec.nombreDeNotes)))
        elif parametres.norm_crossConcTot=='energy + conc':
            self.crossConcordanceTot = np.divide(np.sum(self.spectreConcordanceTot * Prec.spectreConcordanceTot)**(2/(self.nombreDeNotes + Prec.nombreDeNotes)), np.sqrt(self.concordanceTotale *  np.sum(Prec.spectreConcordanceTot)**(2/Prec.nombreDeNotes)))
        elif parametres.norm_crossConcTot=='energy':
            self.crossConcordanceTot = np.sum(self.spectreConcordance * Prec.spectreConcordanceTot)**(2/(self.nombreDeNotes + Prec.nombreDeNotes))/self.energyUniss[0]


    def BaryConcordanceTot(self):
        self.baryConcTot = np.sum(self.spectreConcordanceTot * np.arange(0,16,0.001)) / (self.concordanceTotale ** (self.nombreDeNotes/2))


    def DifBaryConcTot(self,Prec): # Calcule les variations de barycentre
        if type(Prec) is int:
            self.difBaryConcTot = ""
        else:
            self.difBaryConcTot = self.baryConcTot - Prec.baryConcTot


    def SpectreAccord(self):
        self.spectreAccord= np.zeros(16000)
        for i, pitch in enumerate(self.listeHauteursAvecMultiplicite):
            self.spectreAccord= self.spectreAccord + self.spectre(self.frequenceAvecTemperament(pitch))
        self.energy = sum(self.spectreAccord * self.spectreAccord)


    def Context(self):
        mem = parametres.memory_size

        if isinstance(mem,int):
            if parametres.memory_type == 'mean':
                #Construction du vecteur de pondération
                weights = [(1/i**parametres.memory_decr_ponderation) for i in range(1,mem+2)]
                #Moyennage
                l = len(listeSpectresAccords)
                if l<=mem:
                    self.context = np.average(np.asarray(listeSpectresAccords), axis=0, weights=[weights[l-1-i] for i in range(l)])
                else:
                    self.context = np.average(np.asarray(listeSpectresAccords)[(l-1-mem):,:], axis=0, weights=[weights[mem-i] for i in range(mem+1)])
        self.energyContext = sum(self.context * self.context)


    def HarmonicChange(self,Prec):
        if type(Prec) is int:
            self.harmonicChange= ""
        elif parametres.norm_harmChange=='None':
            if parametres.type_harmChange=='absolute': self.harmonicChange = np.sum(np.abs(self.spectreAccord - Prec.spectreAccord))
            else: self.harmonicChange = np.sum(self.spectreAccord - Prec.spectreAccord)
        elif parametres.norm_harmChange=='frame_by_frame':
            if parametres.type_harmChange=='absolute': self.harmonicChange = np.sum(np.abs(self.spectreAccord/np.sqrt(self.energy) - Prec.spectreAccord/np.sqrt(Prec.energy)))
            else: self.harmonicChange = np.sum(self.spectreAccord/np.sqrt(self.energy) - self.spectreAccord/np.sqrt(Prec.energy))
        elif parametres.norm_harmChange=='general':
            if parametres.type_harmChange=='absolute': self.harmonicChange = np.sum(np.abs((self.spectreAccord - Prec.spectreAccord)/np.power(self.energy * Prec.energy, 1/4)))
            else: self.harmonicChange = np.sum((self.spectreAccord - Prec.spectreAccord)/np.power(self.energy * Prec.energy, 1/4))


    def DiffConcordance(self,Prec):
        if type(Prec) is int:
            self.diffConcordance = ""
        else:
            for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
                for j, pitch2 in enumerate(Prec.listeHauteursAvecMultiplicite):
                    if parametres.norm_diffConc=='energy':
                        self.diffConcordance = self.diffConcordance + np.divide(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))), np.sqrt(self.energy * Prec.energyContext))
                    if parametres.norm_diffConc=='energy + conc':
                        self.diffConcordance = self.diffConcordance + np.divide(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))), np.sqrt(self.energy * Prec.energyC)* self.concordance*Prec.concordance)
                    if parametres.norm_diffConc=='first':
                        self.diffConcordance = self.diffConcordance + np.divide(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))), np.sqrt(self.energy * Prec.energy)* Prec.concordance)


    def DiffConcordanceContext(self,Prec):
        if type(Prec) is int:
            self.diffConcordanceContext = ""
        else:
            if parametres.norm_diffConcContext=='energy':
                self.diffConcordanceContext = np.divide(np.sum(self.spectreAccord*Prec.context), np.sqrt(self.energy * Prec.energy))

    def DiffRoughness(self,Prec):
        if type(Prec) is int:
            self.diffRoughness = ""
        else:
            for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
                for j, pitch2 in enumerate(Prec.listeHauteursAvecMultiplicite):
                    for k1 in range(1,len(self.partiels) + 1):
                        for k2 in range(1,len(Prec.partiels) + 1):
                            freq = [Prec.partiels[k2-1] * self.frequenceAvecTemperament(pitch2), self.partiels[k1-1] * self.frequenceAvecTemperament(pitch1)]
                            freq.sort()
                            fmin, fmax = freq[0], freq[1]
                            s = 0.44*(np.log(parametres.β2/parametres.β1)/(parametres.β2-parametres.β1))*(fmax-fmin)/(fmin**(0.477))
                            diss = np.exp(-parametres.β1*s)-np.exp(-parametres.β2*s)
                            if parametres.type_diffDiss=='produit':
                                self.diffRoughness = self.diffRoughness + (self.amplitudes[k1-1] * Prec.amplitudes[k2-1]) * diss

            if parametres.type_diffDiss=='produit':
                self.diffRoughness = self.diffRoughness/np.sqrt(self.energy * Prec.energy)

            n = self.nombreDeNotes
            self.diffRoughness = self.diffRoughness/(n*(n - 1)/2)

    # def DiffRoughnessContext(self,Prec):
    #     def rough(f0,f1):
    #         s = 0.24/(0.021*f0 + 19)
    #         return np.exp(-parametres.β1*s*(f1-f0))-np.exp(-parametres.β2*s*(f1-f0))
    #     if type(Prec) is int:
    #         self.diffRoughnessContext = ""
    #     else:
    #         dic1, dic2 = self.spectreAccord_pic, Prec.context_pic
    #         for f1 in dic1:
    #             for f2 in dic2:
    #                 freq = [f1,f2]
    #                 freq.sort()
    #                 fmin, fmax = freq[0], freq[1]
    #                 self.diffRoughnessContext += (dic1[f1] * dic2[f2]) * rough(fmin,fmax)
    #
    #         self.diffRoughnessContext = self.diffRoughnessContext/np.sqrt(self.energy * Prec.energyContext)



    def HarmonicNovelty(self,Prec):
        #Construction de spectreHarmonicNovelty
        if type(Prec) is int:
            if parametres.type_harmNov == 'dyn':
                self.harmonicNovelty = ""
            else:
                self.spectreHarmonicNovelty = self.spectreAccord
                self.harmonicNovelty = np.divide(np.sum(self.spectreHarmonicNovelty * self.spectreHarmonicNovelty), self.energy)
        else:
            #Construction de spectreHarmonicNovelty
            self.spectreHarmonicNovelty = self.spectreAccord - Prec.context
            for i in range(len(self.spectreAccord)):
                if self.spectreHarmonicNovelty[i] < 0:
                    self.spectreHarmonicNovelty[i]=0

            #Calcul de harmonicNovelty
            self.harmonicNovelty = np.divide(np.sum(self.spectreHarmonicNovelty * self.spectreHarmonicNovelty), self.energy)





# PARAMETRES
type_Temporal = parametres.type_Temporal
type_Normalisation = parametres.type_Normalisation

# CONSTRUCTION DE LA MATRICE POINTS

# Palestrina, AccordsMajeurs, AccordsMineur, Majeur3et4notes, Majeur3et4notes, Accords3Notes, DispoMajeurMineur, Tension
# Cadence3V, Cadence4VMaj, Cadence4Vmin, SuiteAccords
# score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/Exemples/AccordsMineur.musicxml')

# Afficher : Concordance, Roughness, Tension, ConcordanceOrdre3, ConcordanceTotale, crossConcordance, crossConcordanceTot,
#            BaryConc, BaryConcTot, DifBaryConc, DifBaryConcTot

# FONCTIONS

# Fonction qui normalise la matrice Points
def Normalise(Points, liste_timbres_or_scores, dic, type_Normalisation = type_Normalisation):
    # ind_instrument = [dic[instrument]-1 for instrument in liste_timbres_or_scores]
    # Points = Points[ind_instrument]
    if type_Normalisation == 'by timbre':
        max = np.amax(Points, axis = (0,2))
        for descr in range(Points.shape[1]):
            Points[:,descr,:] /= max[descr]
    elif type_Normalisation == 'by curve':
        max = np.amax(Points, axis = 2)
        for timbre in range(Points.shape[0]):
            for descr in range(Points.shape[1]):
                Points[timbre,descr,:] /= max[timbre,descr]
    return Points

# Fonction qui calcule la matrice des écart-types sur tous les timbres
def Dispersion(Points,type_Temporal = type_Temporal):
    if type_Temporal == 'static':
        Disp = np.std(Points,axis = 0)
    elif type_Temporal == 'differential':
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        Disp = np.std(Points_diff,axis = 0)
    Disp_by_descr = np.mean(Disp, axis = 1)
    Disp_by_time = np.linalg.norm(Disp, axis = 0)
    return Disp, Disp_by_descr,Disp_by_time


def Inerties(Points, type_Temporal = type_Temporal):
    if type_Temporal == 'static':
        Inertie_tot = np.std(Points, axis = (0,2))
        Mean = np.mean(Points,axis = 0)
    elif type_Temporal == 'differential':
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        Inertie_tot = np.std(Points_diff, axis = (0,2))
        Mean = np.mean(Points_diff, axis = 0)
    Inertie_inter = np.std(Mean, axis = 1)
    return Inertie_tot, Inertie_inter

# Fonction qui trie les descripteurs en fonction du minimum de dispersion
def MinimizeDispersion(Disp_by_descr, space):
    disp_sorted = np.sort(Disp_by_descr)
    descr_sorted = [space[i] for i in np.argsort(Disp_by_descr)]
    return descr_sorted, disp_sorted

# Fonction qui trie les descripteurs en fonction du minimum de dispersion
def MaximizeSeparation(Inertie_tot, Inertie_inter, space):
    d = len(space)
    sep_matrix = np.zeros((d,d))
    for i in range(1,d):
        for j in range(i):
            sep_matrix[i,j] = np.inner(Inertie_inter[[i,j]], Inertie_inter[[i,j]]) / np.inner(Inertie_tot[[i,j]], Inertie_tot[[i,j]])
    ind = np.unravel_index(np.argmax(sep_matrix, axis=None), sep_matrix.shape)
    return [space[ind[0]], space[ind[1]]]


def Clustered(Points, spacePlot, space, type_Temporal = type_Temporal):
    ind_descr = [space.index(descr) for descr in spacePlot]
    if type_Temporal == 'static':
        Points_sub = Points[:,ind_descr]
    if type_Temporal == 'differential':
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        Points_sub = Points_diff[:,ind_descr]
    disp_traj = np.sum(np.linalg.norm(np.std(Points_sub,axis = 0), axis = 0))
    inertie_inter = np.std(np.mean(Points_sub,axis = 0), axis = 1)
    inertie_tot = np.std(Points_sub, axis = (0,2))
    sep = np.inner(inertie_inter, inertie_inter) / np.inner(inertie_tot, inertie_tot)
    print('Dispersion : {} \nSeparation : {}'.format(disp_traj, sep))


def Visualize(Points, descr, space, liste_timbres_or_scores, dic, type_Temporal = type_Temporal):
    dim1 = space.index(descr[0])
    dim2 = space.index(descr[1])

    # Fonction qui renvoie True si deux listes ont une intersection commune
    def intersect(lst1, lst2):
        inter = False
        i = 0
        while (not inter) & (i<len(lst1)):
            if (lst1[i] in lst2): inter = True
            i += 1
        return inter


    # Détermination de la présence simultanée de descripteurs statiques et dynamiques dans space, et le cas échéant attribution du suffixe 'evolution' aux descr stat
    suff0, suff1 = '',''
    if intersect(space,spaceStat) & intersect(space,spaceDyn):
        if descr[0] in spaceStat: suff0 = ' evolution'
        if descr[1] in spaceStat: suff1 = ' evolution'

    plt.figure(figsize=(8, 7))
    ax = plt.subplot()


    if type_Temporal =='static':
        for track in range(Points.shape[0]):
            if track <= 9: ls = '--'
            else: ls = ':'
            if parametres.visualize_trajectories:
                if parametres.compare_instruments:
                    plt.plot(Points[track,dim1,:].tolist(), Points[track,dim2,:].tolist(), color ='C{}'.format(track),ls = ls,marker = 'o', label = 'Timbre{}:{}'.format(track+1, dic[track]))
                if parametres.compare_scores:
                    plt.plot(Points[track,dim1,:].tolist(), Points[track,dim2,:].tolist(), color ='C{}'.format(track),ls = ls,marker = 'o', label = liste_scores[track])
            if parametres.visualize_time_grouping:
                for t in range(len(Points[track,dim1,:].tolist())):
                    plt.plot(Points[track,dim1,:].tolist()[t], Points[track,dim2,:].tolist()[t], color ='C{}'.format(t), marker = 'o')
            for t in range(len(Points[track,dim1,:].tolist())):
                ax.annotate(' {}'.format(t+1), (Points[track,dim1,:][t], Points[track,dim2,:][t]), color='black')
        if not all(x>=0 for x in Points[:,dim1,:].flatten()):
            plt.vlines(0,np.amin(Points[:,dim2,:]), np.amax(Points[:,dim2,:]), alpha=0.5, linestyle = ':')
        if not all(x>=0 for x in Points[:,dim2,:].flatten()):
            plt.hlines(0,np.amin(Points[:,dim1,:]), np.amax(Points[:,dim1,:]), alpha=0.5, linestyle = ':')
        plt.xlabel(descr[0][0].upper() + descr[0][1:] + suff0)
        plt.ylabel(descr[1][0].upper() + descr[1][1:] + suff1)
        if parametres.compare_instruments: goal, spectr = 'Timbre comparaison', ''
        elif parametres.compare_scores: goal, spectr = 'Score comparaison', '(K={},decr={},sig={})'.format(timbre[0],timbre[1],timbre[2])
        if type_Normalisation == 'by curve':
            plt.title(goal + ', modelised spectrum ' + spectr + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation curve by curve' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')
        else:
            plt.title(goal + ', modelised spectrum ' + spectr + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation on all the curves' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')

    elif type_Temporal =='differential':
        # Construction de la matrice Points_diff
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        for track in range(Points.shape[0]):
            if track <= 9: ls = '--'
            else: ls = ':'
            if parametres.visualize_trajectories:
                if parametres.compare_instruments:
                    plt.plot(Points_diff[track,dim1,:].tolist(), Points_diff[track,dim2,:].tolist(), color ='C{}'.format(track),ls = ls,marker = 'o', label = 'Timbre{}:{}'.format(track+1, dic[track]))
                if parametres.compare_scores:
                    plt.plot(Points_diff[track,dim1,:].tolist(), Points_diff[track,dim2,:].tolist(), color ='C{}'.format(track),ls = ls,marker = 'o', label = liste_scores[track])
            if parametres.visualize_time_grouping:
                for t in range(len(Points_diff[track,dim1,:].tolist())):
                    plt.plot(Points_diff[track,dim1,:].tolist()[t], Points_diff[track,dim2,:].tolist()[t], color ='C{}'.format(t), marker = 'o')
            for t in range(len(Points_diff[track,dim1,:].tolist())):
                ax.annotate(' {}'.format(t+1), (Points_diff[track,dim1,:][t], Points_diff[track,dim2,:][t]), color='black')

        if not all(x>=0 for x in Points_diff[:,dim1,:].flatten()):
            plt.vlines(0,np.amin(Points_diff[:,dim2,:]), np.amax(Points_diff[:,dim2,:]), alpha=0.5, linestyle = ':')
        if not all(x>=0 for x in Points_diff[:,dim2,:].flatten()):
            plt.hlines(0,np.amin(Points_diff[:,dim1,:]), np.amax(Points_diff[:,dim1,:]), alpha=0.5, linestyle = ':')

        plt.xlabel(descr[0][0].upper() + descr[0][1:] + suff0 + ' evolution')
        plt.ylabel(descr[1][0].upper() + descr[1][1:] + suff1 + ' evolution')
        if parametres.compare_instruments: goal, spectr = 'Timbre comparaison', ''
        elif parametres.compare_scores: goal, spectr = 'Score comparaison', '(K={},decr={},sig={})'.format(timbre[0],timbre[1],timbre[2])
        if type_Normalisation == 'by curve':
            plt.title(goal + ', modelised spectrum ' + spectr + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation curve by curve' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')
        else:
            plt.title(goal + ', modelised spectrum ' + spectr + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation on all the curves' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')

    plt.legend(frameon=True, framealpha=0.75)
    plt.show()


spaceStat = ['roughness','concordance','concordanceOrdre3','concordanceTotale']
spaceDyn = ['harmonicChange','harmonicNovelty','diffRoughness','diffConcordance','diffConcordanceContext']


if parametres.one_track:

    # title = 'enum_Norm_3'
    title = 'Test 1 note'
    # space = ['roughness','harmonicity']
    space = ['diffConcordance']
    score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/test1note.musicxml')
    if os.path.exists('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+title+'-score.png'):
        partition = '/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+title+'-score.png'
    else: partition = ''
    l = ListeAccords(score, partition = partition)
    l.HarmonicDescriptors()
    # l.Classement('harmonicity', reverse = True)
    l.getAnnotatedStream(space)
    l.stream.show()
    # l.Affichage(space)
    # axe = 'concordance'
    # liste = l.Liste(axe)
    # # for accord in l.grandeursHarmoniques:
    # #     # if isinstance(getattr(accord, axe),str): pass
    # #     # else: liste.append(getattr(accord, axe))
    # #     axe = 'concordance'
    # #     liste.append(getattr(accord, axe))
    # print(liste)

###### COMPARAISON DES TIMBRES

if parametres.compare_instruments:
    liste_timbres = range(9)
    # dic_timbres[i] = (K, decr, sig)
    dic_timbres = {}
    dic_timbres[0] = (3,1/2,0.01)
    dic_timbres[1] = (3,1,0.01)
    dic_timbres[2] = (3,2,0.01)
    dic_timbres[3] = (7,1/2,0.01)
    dic_timbres[4] = (7,1,0.01)
    dic_timbres[5] = (7,2,0.01)
    dic_timbres[6] = (15,1/2,0.01)
    dic_timbres[7] = (15,1,0.01)
    dic_timbres[8] = (15,2,0.01)


    title = 'CadenceM2'
    score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/ExemplesMusicaux/'+title+'.musicxml')
    space = spaceDyn



    # CONSTRUCTION DE LA MATRICE POINTS
    N = len(liste_timbres)
    Points = []

    # Construction de Points
    for ind in range(N):
        l = ListeAccords(score, dic_timbres[ind])
        l.HarmonicDescriptors()
        Points.append(l.Points(space))
    Points = np.asarray(Points)


    # MAIN
    Points = Normalise(Points, liste_timbres, dic_timbres)
    Disp, Disp_by_descr, Disp_by_time = Dispersion(Points)
    Inertie_tot, Inertie_inter = Inerties(Points)
    descr_sorted, disp_sorted = MinimizeDispersion(Disp_by_descr, space)
    descrs_max_sep = MaximizeSeparation(Inertie_tot, Inertie_inter, space)
    spacePlot = ['harmonicChange', 'diffConcordance']
    # spacePlot = descr_sorted[0:2]
    # spacePlot = descrs_max_sep

    # Visualize(descr = ['concordance','diffConcordance'])
    Clustered(Points, spacePlot, space)
    Visualize(Points, spacePlot, space, liste_timbres, dic_timbres)


if parametres.compare_scores:
    title = 'Cadence_M'
    liste_scores = ['Cadence 1', 'Cadence 2','Cadence 3', 'Cadence 4', 'Cadence 5', 'Cadence 6', 'Cadence 7', 'Cadence 8', 'Cadence 9' ]
    dic_scores = {liste_scores[i]:i+1 for i in range(len(liste_scores))}

    space = spaceDyn

    # CONSTRUCTION DE LA MATRICE POINTS
    N = len(liste_scores)
    Points = []

    # Construction de Points
    timbre = (7,1/2,0.01)
    for ind in range(1,N+1):
        score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Cadence_M{}.musicxml'.format(ind))
        l = ListeAccords(score, timbre)
        l.HarmonicDescriptors()
        Points.append(l.Points(space))
    Points = np.asarray(Points)

    # MAIN
    Points = Normalise(Points, liste_scores, dic_scores)
    Disp, Disp_by_descr, Disp_by_time = Dispersion(Points)
    Inertie_tot, Inertie_inter = Inerties(Points)
    descr_sorted, disp_sorted = MinimizeDispersion(Disp_by_descr, space)
    descrs_max_sep = MaximizeSeparation(Inertie_tot, Inertie_inter, space)
    spacePlot = ['harmonicChange', 'diffConcordance']
    # spacePlot = descr_sorted[0:2]
    # spacePlot = descrs_max_sep

    # Visualize(descr = ['concordance','diffConcordance'])
    Clustered(Points, spacePlot, space)
    Visualize(Points, spacePlot, space, liste_scores, dic_scores)



######## COMPARAISON DES DESCRIPTEURS

if parametres.compare_descriptors:

    # CONSTRUCTION DE LA DATA FRAME
    N = len(parametres.timbres)
    l = ListeAccords(score)
    l.HarmonicDescriptors()
    Puncts = l.Points(space)
    # On prend en compte tous les timbres
    for ind in range(1,N):
        l = ListeAccords(score, ind)
        l.HarmonicDescriptors()
        Puncts = np.concatenate((Puncts, l.Points(space)),axis=1)

    Puncts = Puncts.T

    # Normalisation
    for descr in range(Puncts.shape[1]):
        Puncts[:,descr] /= np.amax(Puncts[:,descr])

    df = pd.DataFrame(Puncts,
                        index = ['Time{}'.format(i) for i in range(Puncts.shape[0])],
                        columns = space)
    print(df)


    # MATRICE DE CORRELATION
    if parametres.correlation:
        corrMatrix = df.corr()
        # print(corrMatrix)

        ax = sns.heatmap(
            corrMatrix, annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
            )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')

        plt.show()

    if parametres.pca:
        x = StandardScaler().fit_transform(df)
        print(type(x))
        print(type(df))
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2']
                 , index = ['Time{}'.format(i) for i in range(Puncts.shape[0])])

        plt.figure()
        plt.plot(principalComponents[:,0],principalComponents[:,1],'ob')
        for i in range(len(principalComponents[:,0])):
            plt.text(principalComponents[i,0],principalComponents[i,1],'Acc{}'.format(i+1))
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()
