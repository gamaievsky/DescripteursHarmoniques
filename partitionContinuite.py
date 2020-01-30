
'''
Etant donnée une liste d'accords, les trie par ordre de concordance, consonance, tension et concordance totale,
en affichant en dessous les valeurs. Prend en entrée dans le fichier paramètre deux listes de même taille : partiels,
qui contient l'emplacement des partiels (éventuellement inharmoniques),  et amplitudes, avec leurs amplitudes
respectives
'''

#from mimetypes import init

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from operator import itemgetter, attrgetter
from music21 import *
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


    def __init__(self, stream, ind = 0):

        self.stream = stream
        self.tree = tree.fromStream.asTimespans(stream, flatten=True,classList=(note.Note, chord.Chord))
        if parametres.mode=='manual':
            self.partiels = parametres.partiels
            self.amplitudes = parametres.amplitudes
            self.sig = parametres.sig
        else:
            self.partiels = []
            self.amplitudes = []
            K = parametres.timbres[ind][0]
            decr = parametres.timbres[ind][1]
            self.sig = parametres.timbres[ind][2]
            for i in range(K):
                self.partiels.append(i+1)
                self.amplitudes.append(1/(i+1)**decr)


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


    def  spectre(self,f0):

        '''Cette methode va etre appelee dans la classe Accord, mais elle est definie ici car le seul attribut d'objet
         qui en est parametre est l'instrument'''

        n = np.arange(0,16,0.001)
        S = np.zeros(np.shape(n))
        for i in range(1, len(self.partiels) + 1):
            S = S + (self.amplitudes[i-1]) * np.exp(-(n - np.log2(self.partiels[i-1] * f0))**2 / (2 * self.sig**2))
        return S


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
            v = Accord(verticality, self.partiels, self.amplitudes, self.sig, self.energyUniss, self.temperament,self.noteDeReferencePourLeTunning)
            if verticality.bassTimespan!=None :
                v.identifiant = verticality.bassTimespan.element.id

            v.ListeHauteursAvecMultiplicite()
            v.NombreDeNotes()

            if v.nombreDeNotes>=2:
                v.SpectreAccord()
                listeSpectresAccords.append(v.spectreAccord)
                v.Context()
                v.Roughness()
                v.SpectreConcordance()
                v.Concordance()
                v.CrossConcordance(Prec)
                v.HarmonicChange(Prec)
                v.HarmonicNovelty(Prec)
                v.DiffConcordance(Prec)
                v.DiffConcordanceContext(Prec)
                v.DiffRoughness(Prec)

                if v.nombreDeNotes>=3:
                    v.ConcordanceOrdre3()
                    v.SpectreConcordanceTot()
                    v.ConcordanceTotale()
                    v.CrossConcordanceTot(Prec)

            Prec = v
            self.grandeursHarmoniques.append(v)


    def getAnnotatedStream(self, space = ['concordance'], write_name = True):
        r=0
        for gH in self.grandeursHarmoniques:
            if gH.verticality.bassTimespan != None :
                element = gH.verticality.bassTimespan.element
                if element.isNote or element.isChord:
                    dataString = ""

                for descr in space:
                    #Calibrage
                    mult = 1
                    if descr in ['concordanceOrdre3','concordanceTotale']: mult = 10
                    elif descr in ['crossConcordance','crossConcordanceTot','concordance']: mult = 100
                    if dataString != '': dataString + " "
                    #Descripteurs différentiels
                    if descr in ['crossConcordance','crossConcordanceTot','difBaryConc','difBaryConcTot']:
                        if type(getattr(gH,descr)) != str:
                            dataString = dataString + "-" + str(round(mult * getattr(gH,descr),2))
                    #Descripteurs statiques
                    else: dataString = dataString + str(round(mult * getattr(gH,descr),2))

                #Rajout du nom du descripteur
                if r == 0:
                    if write_name:
                        dataString = dataString  + "\n" + space[0][0].upper() + space[0][1:]
                        r=1
                #Assignement
                element.lyric = dataString

        return tree.toStream.partwise(self.tree, self.stream)


    def Liste(self, axe = 'concordance'):
        liste = []
        for accord in self.grandeursHarmoniques:
            if isinstance(getattr(accord, axe),str): pass
            else: liste.append(getattr(accord, axe))
        return liste


    def Classement(self, descr):
      s = stream.Measure()
      ts1 = meter.TimeSignature('C')
      s.insert(0, ts1)
      s.insert(0, clef.TrebleClef())

      self.stream.lyrics = {}
      self.getAnnotatedStream([descr], write_name = False)
      self.grandeursHarmoniques.sort(key=attrgetter(descr), reverse=True)
      for gH in self.grandeursHarmoniques:
          element = gH.verticality.bassTimespan.element
          s.insert(-1, element)
      s[0].addLyric(descr[0].upper() + descr[1:] + ' triée')
      s.show()
      del s[0].lyrics[1]

    '''Fonction qui regroupe dans un tableau de dimension d*T ou d*(T-1) les valeurs des d descripteurs sur une séquence de T accords
    '''
    def Points(self,space):
        d = len(space)
        L = []
        for descr in space:
            L.append(self.Liste(descr))
        T = min(map(len,L))
        for i in range(d):
            if len(L[i])>T: L[i].pop(0)

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


        # plt.title(title)
        s = int(parametres.aff_score)


        for k, descr in enumerate(space):
            l = self.Liste(descr)
            if "" in l: l.remove("")
            plt.subplot(dim+s, 1, k+1+s)
            plt.vlines(times, min(l), max(l), color='k', alpha=0.9, linestyle='--')
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

    def __init__(self, verticality, partiels, amplitudes, sig, energyUniss, temperament, noteDeReferencePourLeTunning):

        self.partiels = partiels
        self.amplitudes = amplitudes
        self.sig = sig
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

    def Roughness(self):

         for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
             for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                 if (i<j):
                     for k1 in range(1,len(self.partiels) + 1):
                         for k2 in range(1,len(self.partiels) + 1):
                             freq = [self.partiels[k2-1] * self.frequenceAvecTemperament(pitch2), self.partiels[k1-1] * self.frequenceAvecTemperament(pitch1)]
                             freq.sort()
                             fmin, fmax = freq[0], freq[1]
                             s = 0.44*(np.log(parametres.β2/parametres.β1)/(parametres.β2-parametres.β1))*(fmax-fmin)/(fmin**(0.477))
                             diss = np.exp(-parametres.β1*s)-np.exp(-parametres.β2*s)
                             if parametres.type_diss=='produit':
                                 self.roughness = self.roughness + (self.amplitudes[k1-1] * self.amplitudes[k2-1]) * diss
                             elif parametres.type_diss == 'minimum':
                                 self.roughness = self.roughness + min(self.amplitudes[k1-1],self.amplitudes[k2-1]) * diss

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
                         for k1 in range(1,len(self.partiels) + 1):
                             for k2 in range(1,len(self.partiels) + 1):
                                 for k3 in range(1,len(self.partiels) + 1):
                                     x = np.log2((self.partiels[k2-1] * self.frequenceAvecTemperament(pitch2)) / (self.partiels[k1-1] * self.frequenceAvecTemperament(pitch1)))
                                     y = np.log2((self.partiels[k3-1] * self.frequenceAvecTemperament(pitch3)) / (self.partiels[k2-1] * self.frequenceAvecTemperament(pitch2)))
                                     z = x + y
                                     X = abs(x)
                                     Y = abs(y)
                                     Z = abs(z)
                                     a = 0.6
                                     diff = [abs(X-Y), abs(X-Z), abs(Y-Z)]
                                     diff.remove(max(diff))
                                     contrib = abs(diff[1]-diff[0])
                                     self.tension = self.tension + (self.amplitudes[k1-1] * self.amplitudes[k2-1] * self.amplitudes[k3-1]) * (np.exp(-(12*(contrib)/a)**2))

                                     #self.tension = self.tension + (self.amplitudes[k1-1] * self.amplitudes[k2-1] * self.amplitudes[k3-1]) * max(np.exp(-(12*(X-Y)/a)**2) , np.exp(-(12*(Y-Z)/a)**2) , np.exp(-(12*(X-Z)/a)**2))

         n = self.nombreDeNotes
         self.tension = self.tension/(self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)

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

        """ Normalisation logarithmique, de maniere a rendre egales les concordances des unissons de n notes"""
        self.concordance = np.sum(self.spectreConcordance)
        self.concordance = self.concordance/(self.nombreDeNotes*(self.nombreDeNotes - 1)/2)

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
        self.baryConc = np.sum(self.spectreConcordance * np.arange(0,16,0.001)) / (self.concordance * (self.nombreDeNotes*(self.nombreDeNotes - 1)/2))


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
        self.energyContext = sum(self.spectreAccord * self.spectreAccord)


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










title = 'CadenceM2'
# Palestrina, AccordsMajeurs, AccordsMineur, Majeur3et4notes, Majeur3et4notes, Accords3Notes, DispoMajeurMineur, Tension
# Cadence3V, Cadence4VMaj, Cadence4Vmin, SuiteAccords
score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/ExemplesMusicaux/'+title+'.musicxml')
#score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/Exemples/AccordsMineur.musicxml')

# Afficher : Concordance, Roughness, Tension, ConcordanceOrdre3, ConcordanceTotale, crossConcordance, crossConcordanceTot,
#            BaryConc, BaryConcTot, DifBaryConc, DifBaryConcTot
spaceStat = ['roughness','concordance','concordanceOrdre3','concordanceTotale']
spaceDyn = ['harmonicChange','harmonicNovelty','diffRoughness','diffConcordance','diffConcordanceContext']
space = spaceStat #+ spaceDyn


###### COMPARAISON DES TIMBRES

# CONSTRUCTION DE LA MATRICE POINTS
N = len(parametres.timbres)
Points = []

for ind in range(N):
    l = ListeAccords(score, ind)
    l.HarmonicDescriptors()
    Points.append(l.Points(space))
Points = np.asarray(Points)

# PARAMETRES
type_Temporal = 'static' #'static', 'differential'
type_Normalisation = 'by timbre' #'by curve', 'by timbre'

# FONCTIONS

# Fonction qui normalise la matrice Points
def Normalise(Points = Points, type_Normalisation = type_Normalisation):
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
def Dispersion(Points = Points,type_Temporal = type_Temporal):
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

def MinimizeDispersion(Disp_by_descr, space = space):
    disp_sorted = np.sort(Disp_by_descr)
    descr_sorted = [space[i] for i in np.argsort(Disp_by_descr)]
    return descr_sorted, disp_sorted

def Visualize(Points = Points, space = space, descr = space[0:2], type_Temporal = type_Temporal):
    dim1 = space.index(descr[0])
    dim2 = space.index(descr[1])

    plt.figure(figsize=(8, 7))
    ax = plt.subplot()

    for timbre in range(Points.shape[0]):
        if type_Temporal =='static':
            plt.plot(Points[timbre,dim1,:].tolist(), Points[timbre,dim2,:].tolist(), color ='C{}'.format(timbre),ls = '--',marker = 'o', label = 'timbre{}: ({},{},{})'.format(timbre+1, parametres.timbres[timbre][0],parametres.timbres[timbre][1],parametres.timbres[timbre][2]))
            for t in range(len(Points[timbre,dim1,:].tolist())):
                ax.annotate(' {}'.format(t+1), (Points[timbre,dim1,:][t], Points[timbre,dim2,:][t]), color='black')
            plt.xlabel(descr[0][0].upper() + descr[0][1:])
            plt.ylabel(descr[1][0].upper() + descr[1][1:])
            plt.title(title + ' (' + descr[0][0].upper() + descr[0][1:] + ', ' + descr[1][0].upper() + descr[1][1:] + ')\n' + 'Normalisation ' + type_Normalisation[3:] + ' ' + type_Normalisation + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')

        elif type_Temporal =='differential':
            Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
            for i in range(Points.shape[2]-1):
                Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
            plt.plot(Points_diff[timbre,dim1,:].tolist(), Points_diff[timbre,dim2,:].tolist(), color ='C{}'.format(timbre),ls = '--',marker = 'o', label = 'timbre{}: ({},{},{})'.format(timbre+1, parametres.timbres[timbre][0],parametres.timbres[timbre][1],parametres.timbres[timbre][2]))
            for t in range(len(Points_diff[timbre,dim1,:].tolist())):
                ax.annotate(' {}'.format(t+1), (Points_diff[timbre,dim1,:][t], Points_diff[timbre,dim2,:][t]), color='black')
            plt.xlabel(descr[0][0].upper() + descr[0][1:] + ' evolution')
            plt.ylabel(descr[1][0].upper() + descr[1][1:] + ' evolution')
            plt.title(title + ' (' + descr[0][0].upper() + descr[0][1:] + ', ' + descr[1][0].upper() + descr[1][1:] + ')\n' + 'Normalisation ' + type_Normalisation[3:] + ' ' + type_Normalisation + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')

    plt.legend(frameon=True, framealpha=0.75)
    plt.show()

# MAIN
Points = Normalise()
Disp, Disp_by_descr,Disp_by_time = Dispersion()
descr_sorted, disp_sorted = MinimizeDispersion(Disp_by_descr)
# Visualize(type = 'static', descr = descr_sorted[0:2])
Visualize(descr = ['concordanceTotale','roughness'])


######## COMPARAISON DES DESCRIPTEURS







# l.Classement('concordance')
# l.getAnnotatedStream(['concordance'])
# l.stream.show()
# l.Affichage(space)
