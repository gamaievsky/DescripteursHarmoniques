import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle
import sys
import os


from music21 import *
from operator import itemgetter, attrgetter
import parametres
import tunings
listeSpectresAccords = []
listeSpectresAccords_pic = []


################ LES CLASSES ##################

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


    def __init__(self, stream, instr = parametres.timbre_def, memory_size = parametres.memory_size,memory_decr_ponderation = parametres.memory_decr_ponderation, partition = ''):

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
        self.partition = partition
        self.memory_size = memory_size
        self.memory_decr_ponderation = memory_decr_ponderation



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
            Σ = 2.0
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
            Σ = 2.0

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


    def frequenceAvecTemperament(self,pitch1):

        """Fonction qui prend en entree un pitch pour renvoyer une frequence, en tenant compte du temperament"""

        pitchRef = pitch.Pitch(self.noteDeReferencePourLeTunning)
        pitch1.microtone = self.temperament[(pitch1.pitchClass - pitchRef.pitchClass)%12] - 100*((pitch1.pitchClass - pitchRef.pitchClass)%12)

        return (pitch1.frequency)


    def HarmonicDescriptors (self):

        ''' Transforme chaque verticalite en objet Accord, calcule la concordance, la coherence et les concordances multiples, et stocke les resultats
        sous forme de liste d'Accords"
        '''

        Prec = 0
        compte = 1
        global listeSpectresAccords, listeSpectresAccords_pic
        listeSpectresAccords = []
        listeSpectresAccords_pic = []

        for verticality in self.tree.iterateVerticalities():
            v = Accord(verticality, self.partiels, self.amplitudes, self.K, self.sig, self.decr,self.memory_size, self.memory_decr_ponderation, self.temperament,self.noteDeReferencePourLeTunning)
            if verticality.bassTimespan!=None :
                v.identifiant = verticality.bassTimespan.element.id

            v.ListeHauteursAvecMultiplicite()
            v.NombreDeNotes()

            if v.nombreDeNotes>=1:
                v.SpectreAccord()
                listeSpectresAccords.append(v.spectreAccord)
                listeSpectresAccords_pic.append(v.spectreAccord_pic)
                v.Context()
                v.HarmonicChange(Prec)
                v.DiffConcordanceContext(Prec)
                v.DiffRoughnessContext(Prec)

            Prec = v
            self.grandeursHarmoniques.append(v)
            # print('Accord {}: OK'.format(compte))
            compte += 1


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
            mmin = min(liste[j])
            if (m-mmin) != 0:
                liste[j] = [(100.0/(m-mmin))*(val-mmin) for val in liste[j]]
        print(liste)
        for gH in self.grandeursHarmoniques:
            if gH.verticality.bassTimespan != None :
                element = gH.verticality.bassTimespan.element
                if element.isNote or element.isChord:
                    dataString = ""

                for d, descr in enumerate(space):
                    if dataString != '': dataString + " "
                    #Descripteurs différentiels
                    if descr in ['crossConcordance','crossConcordanceTot','difBaryConc','difBaryConcTot']:
                        if type(getattr(gH,descr)) != str:
                            # dataString = dataString + "-" + str(round(liste[d][i],1))
                            dataString = dataString + "-" + str(int(liste[d][i]))

                    #Descripteurs statiques
                    else:
                        # dataString = dataString + str(round(liste[d][i],1))
                        dataString = dataString + str(int(liste[d][i-1]))

                #Rajout du nom du descripteur
                if r == 0:
                    if write_name:
                        if parametres.shepard: shep = 'Shepard, '
                        else: shep = 'no Shepard, '
                        dataString = dataString  + "\n" + space[0][0].upper() + space[0][1:] +'\n'+ shep + 'K : {}, σ : {}, decr : {}'.format(self.K,self.sig,self.decr)
                        r=1

                #Assignement
                # element.lyric = dataString
                # element.lyric = element.forteClass
                element.lyric = element.intervalVector
            i+=1
        return tree.toStream.partwise(self.tree, self.stream)



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


class Accord(ListeAccords):
    '''
    Classe qui traite les verticalites en heritant de l'instrument et de la methode spectre de la classe ListeAccords,
    et ayant comme attributs supplementaires les grandeurs lies a la concordance
    Faiblesse pour l'instant : l'arbre de la classe mere est vide, un attribut 'verticality' vient le remplacer
    '''

    def __init__(self, verticality, partiels, amplitudes,K, sig, decr, memory_size, memory_decr_ponderation, temperament, noteDeReferencePourLeTunning):

        self.partiels = partiels
        self.amplitudes = amplitudes
        self.K = K
        self.sig = sig
        self.decr = decr
        self.memory_size = memory_size
        self.memory_decr_ponderation = memory_decr_ponderation
        self.temperament = temperament
        self.noteDeReferencePourLeTunning = noteDeReferencePourLeTunning
        self.energy = 0
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
        self.spectreAccord_pic = 0
        self.context = 0
        self.context_pic = {}
        self.energyContext = 0
        self.harmonicChange = 0
        self.diffConcordance = 0
        self.diffConcordanceContext = 0
        self.diffRoughnessContext = 0
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


    def SpectreAccord(self):
        self.spectreAccord= np.zeros(16000)
        self.spectreAccord_pic = {}
        for i, pitch in enumerate(self.listeHauteursAvecMultiplicite):
            self.spectreAccord= self.spectreAccord + self.spectre(self.frequenceAvecTemperament(pitch))
            dic = self.spectre_pic(self.frequenceAvecTemperament(pitch))
            for f in dic:
                if f not in self.spectreAccord_pic: self.spectreAccord_pic[f] = dic[f]
                else: self.spectreAccord_pic[f] += dic[f]
        self.energy = sum(self.spectreAccord * self.spectreAccord)


    def Context(self):
        mem = self.memory_size

        if isinstance(mem,int):
            if parametres.memory_type == 'mean':
                #Construction du vecteur de pondération
                weights = [(1/i**self.memory_decr_ponderation) for i in range(1,mem+2)]
                #Moyennage pour c
                l = len(listeSpectresAccords)
                if l<=mem:
                    self.context = np.average(np.asarray(listeSpectresAccords), axis=0, weights=[weights[l-1-i] for i in range(l)])
                    for i in range(l):
                        for f in listeSpectresAccords_pic[-i-1]:
                            if f not in self.context_pic: self.context_pic[f] = weights[i]*listeSpectresAccords_pic[-i-1][f]
                            else: self.context_pic[f] += weights[i]*listeSpectresAccords_pic[-i-1][f]
                    sum = np.sum(weights[:l])
                    for f in self.context_pic:
                        self.context_pic[f] /= sum

                else:
                    self.context = np.average(np.asarray(listeSpectresAccords)[(l-1-mem):,:], axis=0, weights=[weights[mem-i] for i in range(mem+1)])
                    for i in range(mem+1):
                        for f in listeSpectresAccords_pic[-i-1]:
                            if f not in self.context_pic: self.context_pic[f] = weights[i]*listeSpectresAccords_pic[-i-1][f]
                            else: self.context_pic[f] += weights[i]*listeSpectresAccords_pic[-i-1][f]
                    somme = np.sum(weights[:(mem+1)])
                    for f in self.context_pic:
                        self.context_pic[f] /= somme

        self.energyContext = LA.norm(self.context,2)**2


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
            if parametres.type_harmChange=='absolute': self.harmonicChange = np.sum(np.abs(self.spectreAccord - Prec.context)) / (self.energy * Prec.energy)**(1./4)
            else: self.harmonicChange = np.sum((self.spectreAccord - Prec.spectreAccord)/np.power(self.energy * Prec.energy, 1./4))


    def DiffConcordanceContext(self,Prec):
        if type(Prec) is int:
            self.diffConcordanceContext = ""
        else:
            if parametres.norm_diffConcContext=='energy':
                self.diffConcordanceContext = np.divide(np.sum(self.spectreAccord*Prec.context), np.sqrt(self.energy * Prec.energyContext))

    def DiffRoughnessContext(self,Prec):
        def rough(f0,f1):
            s = 0.24/(0.021*f0 + 19)
            return np.exp(-parametres.β1*s*(f1-f0))-np.exp(-parametres.β2*s*(f1-f0))
        if type(Prec) is int:
            self.diffRoughnessContext = ""
        else:
            dic1, dic2 = self.spectreAccord_pic, Prec.context_pic
            for f1 in dic1:
                for f2 in dic2:
                    freq = [f1,f2]
                    freq.sort()
                    fmin, fmax = freq[0], freq[1]
                    self.diffRoughnessContext += (dic1[f1] * dic2[f2]) * rough(fmin,fmax)

            self.diffRoughnessContext = self.diffRoughnessContext/np.sqrt(self.energy * Prec.energyContext)


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


################ CALCUL DES DESCRIPTEURS ##################


# score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/ListeAccordsNiv1-score.musicxml')
#
# l = ListeAccords(score)
# l.HarmonicDescriptors()
