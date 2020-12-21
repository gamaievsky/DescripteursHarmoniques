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


################ LES CLASSES ############################

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


    def __init__(self, stream, instr = parametres.timbre_def, partition = '', classe = None):

        self.stream = stream
        self.tree = tree.fromStream.asTimespans(stream, flatten=True,classList=(note.Note, chord.Chord))
        if parametres.mode=='manual':
            self.partiels = parametres.partiels
            self.amplitudes = parametres.amplitudes
            self.σ = parametres.σ
        else:
            self.partiels = []
            self.amplitudes = []
            self.K = instr[0]
            self.decr = instr[1]
            self.σ = instr[2]
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
        self.classe = classe
        self.partition = partition


    def spectre(self,f0,harmonicity = False):

        '''Cette methode va etre appelee dans la classe Accord, mais elle est definie ici car le seul attribut d'objet
         qui en est parametre est l'instrument'''
        n = np.arange(0,16,0.001)
        S = np.zeros(np.shape(n))

        if not parametres.shepard:
            for i in range(1, len(self.partiels) + 1):
                S = S + (self.amplitudes[i-1]) * np.exp(-(n - np.log2(self.partiels[i-1] * f0))**2 / (2 * self.σ**2))

        else:
            def repr_classe(f0,fmin,fmax):
                if fmin <= f0 < fmax: return f0
                elif f0 < fmin: return repr_classe(2*f0,fmin,fmax)
                elif f0 >= fmax: return repr_classe(f0/2,fmin,fmax)
            f0 = repr_classe(f0,261.0,522.0)
            p0 = np.log2(261)
            Σ = 2.0
            E = np.exp(-(n - p0)**2 / (2 * Σ**2))

            if not harmonicity:
                for k in range(1,self.K+1):
                    f = repr_classe(k*f0,261.0,522.0)
                    p = np.log2(f)
                    for i in range(-8,8):
                        if 0 < p +i < 16:
                            S += (1/k**self.decr) * np.exp(-(n - (p+i))**2 / (2 * self.σ**2))
            else:
                for k in range(1,self.K+10+1):
                    f = repr_classe(k*f0,261.0,522.0)
                    p = np.log2(f)
                    for i in range(-8,8):
                        if 0 < p +i < 16:
                            S += np.exp(-(n - (p+i))**2 / (2 * self.σ**2))
            S *= E
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

        for verticality in self.tree.iterateVerticalities():
            v = Accord(verticality, self.partiels, self.amplitudes, self.K, self.σ, self.decr, self.temperament,self.noteDeReferencePourLeTunning)
            if verticality.bassTimespan!=None :
                v.identifiant = verticality.bassTimespan.element.id

            v.ListeHauteursAvecMultiplicite()
            v.NombreDeNotes()

            if self.classe == 'prime':
                if v.nombreDeNotes>=2:
                    v.SpectreAccord()
                    listeSpectresAccords.append(v.spectreAccord)
                    v.Roughness()
                    v.SpectreConcordance()
                    v.Concordance()
            elif self.classe == 'normal':
                if v.nombreDeNotes>=2:
                    v.SpectreAccord()
                    listeSpectresAccords.append(v.spectreAccord)
                    v.Harmonicity()
                    if v.nombreDeNotes>=3:
                        v.ConcordanceOrdre3()
                        v.SpectreConcordanceTot()
                        v.ConcordanceTotale()
                        v.Tension()
            else:
                if v.nombreDeNotes>=2:
                    v.SpectreAccord()
                    listeSpectresAccords.append(v.spectreAccord)
                    # v.Context()
                    v.Roughness()
                    v.SpectreConcordance()
                    v.Concordance()
                    # v.Harmonicity()
                    # v.HarmonicChange(Prec)
                    # v.HarmonicNovelty(Prec)
                    # v.DiffConcordance(Prec)
                    # v.DiffConcordanceContext(Prec)
                    # v.DiffRoughness(Prec)

                    # if v.nombreDeNotes>=3:
                    #     v.ConcordanceOrdre3()
                    #     v.SpectreConcordanceTot()
                    #     v.ConcordanceTotale()
                    #     v.Tension()
                #
                # Prec = v


            self.grandeursHarmoniques.append(v)
            print('Accord {}: OK'.format(compte))
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
            if m != 0:
                liste[j] = [(100/m)*val for val in liste[j]]
        for gH in self.grandeursHarmoniques:
            if gH.verticality.bassTimespan != None :
                element = gH.verticality.bassTimespan.element
                if element.isNote or element.isChord:
                    dataString = ""

                for d, descr in enumerate(space):
                    if dataString != '': dataString + " "
                    # dataString = dataString + str(round(liste[d][i],1))
                    dataString = dataString + str(int(liste[d][i]))

                #Rajout du nom du descripteur
                if r == 0:
                    if write_name:
                        if parametres.shepard: shep = 'Shepard, '
                        else: shep = 'no Shepard, '
                        dataString = dataString  + "\n" + space[0][0].upper() + space[0][1:] +'\n'+ shep + 'K : {}, σ : {}, decr : {}'.format(self.K,self.σ,self.decr)
                        r=1

                #Assignement
                # element.lyric = dataString
                # element.lyric = element.forteClass
                element.lyric = element.intervalVector
            i+=1
        return tree.toStream.partwise(self.tree, self.stream)

    def getAnnotatedChords(self, type = 'intervalVector'):
        for gH in self.grandeursHarmoniques:
            element = gH.verticality.bassTimespan.element
            if type == 'intervalVector':
                element.duration = duration.Duration(4)
                element.lyric = element.forteClass
                print(element.lyric)

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
        s[0].addLyric(descr[0].upper() + descr[1:] + ' triée' + '\n'+ shep + 'K : {}, σ : {}, decr : {}'.format(self.K,self.σ,self.decr))
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
        plt.title('K : {}, σ : {}, decr : {}'.format(self.K,self.σ,self.decr))

        # plt.title(title)
        s = int(parametres.aff_score and (len(self.partition) != 0))

        if s:
            img=mpimg.imread(self.partition)
            score = plt.subplot(dim+s,1,1)
            plt.axis('off')
            score.imshow(img)
            plt.title('K : {}, σ : {}, decr : {}'.format(self.K,self.σ,self.decr))


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

    def __init__(self, verticality, partiels, amplitudes,K, σ, decr, temperament, noteDeReferencePourLeTunning):

        self.partiels = partiels
        self.amplitudes = amplitudes
        self.K = K
        self.σ = σ
        self.decr = decr
        self.temperament = temperament
        self.noteDeReferencePourLeTunning = noteDeReferencePourLeTunning
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

    def Roughness(self):
        def rough(f0,f1):
            s = 0.24/(0.021*f0 + 19)
            return np.exp(-parametres.β1*s*(f1-f0))-np.exp(-parametres.β2*s*(f1-f0))
        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                if (i<j):
                    dic1, dic2 = self.spectre_pic(self.frequenceAvecTemperament(pitch1)), self.spectre_pic(self.frequenceAvecTemperament(pitch2))
                    for f1 in dic1:
                        for f2 in dic2:
                            freq = sorted([f1,f2])
                            self.roughness += (dic1[f1]*dic2[f2]) * rough(freq[0],freq[1])

        n = self.nombreDeNotes
        self.roughness *= (2*n/(n - 1))/self.energy


    def Tension(self):
        def tens(f0,f1,f2):
            x = np.log2(f1/f0)
            y = np.log2(f2/f1)
            if 12*(abs(y-x)) < 2:
                    return  np.exp(-(12*(abs(y-x))/parametres.δ)**2)
            else: return 0
        dic_contrib = {}
        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                for l, pitch3 in enumerate(self.listeHauteursAvecMultiplicite):
                    if (i<j<l):
                        dic1, dic2, dic3 = self.spectre_pic(self.frequenceAvecTemperament(pitch1)), self.spectre_pic(self.frequenceAvecTemperament(pitch2)), self.spectre_pic(self.frequenceAvecTemperament(pitch3))
                        for f1 in dic1:
                            for f2 in dic2:
                                for f3 in dic3:
                                    freq = sorted([f1,f2,f3])
                                    freq = tuple(freq)
                                    if freq in dic_contrib:
                                        self.tension += (dic1[f1] * dic2[f2] * dic3[f3]) * dic_contrib[freq]
                                    else:
                                        x = np.log2(freq[1] / freq[0])
                                        y = np.log2( freq[2] / freq[1])
                                        if 12*(abs(y-x)) < 2:
                                                dic_contrib[freq] = np.exp(-(12*(abs(y-x))/parametres.δ)**2)
                                                self.tension += (dic1[f1] * dic2[f2] * dic3[f3]) * dic_contrib[freq]


        n = self.nombreDeNotes
        self.concordanceOrdre3 *= (n**3 / (n *(n-1)*(n-2)/6)) / LA.norm(self.spectreAccord,ord=3)**3


    def Harmonicity(self):
        f0 = 261.0
        corr = np.correlate(self.spectreAccord, self.spectre(f0), 'full')
        self.harmonicity = np.nanmax(corr) / LA.norm(self.spectreAccord)


    def ConcordanceOrdre3(self):

        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                for k, pitch3 in enumerate(self.listeHauteursAvecMultiplicite):
                    if (i<j<k):
                        self.concordanceOrdre3 += np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))*self.spectre(self.frequenceAvecTemperament(pitch3)))
        n = self.nombreDeNotes
        self.concordanceOrdre3 *= (n**3 / (n *(n-1)*(n-2)/6)) / LA.norm(self.spectreAccord,ord=3)**3
        # self.concordanceOrdre3 = self.concordanceOrdre3**(1./3)

    def SpectreConcordance(self):
        self.spectreConcordance = np.zeros(16000)
        for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
            for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
                if (i<j):
                    self.spectreConcordance = self.spectreConcordance + self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))

    def Concordance(self):

        """ Normalisation logarithmique, de maniere a rendre egales les concordances des unissons de n notes"""
        self.concordance = np.sum(self.spectreConcordance)
        # self.concordance /= (self.nombreDeNotes*(self.nombreDeNotes - 1)/2)
        n = self.nombreDeNotes
        self.concordance *= (n/((n - 1)/2))/self.energy


    def SpectreConcordanceTot(self):
        self.spectreConcordanceTot = np.ones(16000)
        for pitch in self.listeHauteursAvecMultiplicite:
                self.spectreConcordanceTot = self.spectreConcordanceTot * self.spectre(self.frequenceAvecTemperament(pitch))


    def ConcordanceTotale(self):
        S = np.ones(16000)
        for pitch in self.listeHauteursAvecMultiplicite:
            S = S*self.spectre(self.frequenceAvecTemperament(pitch))
        n= self.nombreDeNotes
        self.concordanceTotale = (n**n / LA.norm(self.spectreAccord,ord = n)**n) * np.sum(self.spectreConcordanceTot)
        # self.concordanceTotale = self.concordanceTotale**(1./n)


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


################ CALCUL DES DESCRIPTEURS ##################

# Il suffit de rentrer le timbre souhaite dans le fichier paramètres (timbre-def)

with open('Dic_id.pkl', 'rb') as f:
    Dic_id = pickle.load(f)

with open('Dic_Harm.pkl', 'rb') as f:
    Dic_Harm = pickle.load(f)

# Fonction qui ajoute un timbre au dictionnaire des valeurs de descripteurs
def AjoutTimbre(K,decr,σ):
    # Classes premières

    score1 = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/ListeAccordsNiv1-score.musicxml')
    l1 = ListeAccords(score1,instr = (K,decr,σ), classe = 'prime')
    l1.HarmonicDescriptors()

    if (K,decr,σ) not in Dic_Harm:
        Dic_Harm[(K,decr,σ)] = {'K':K,'decr':decr,'σ':σ}

    space1 = ['concordance','roughness']
    for descr in space1:
        listeDescr = l1.Liste(descr)
        maxDescr = max(listeDescr)
        listeDescr = [100*float(l)/maxDescr for l in listeDescr]
        Dic_Harm[(K,decr,σ)][descr] = {id: listeDescr[id-1]  for id in range(1,len(listeDescr)+1)}

    # Classes normales

    score2 = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/ListeAccordsNiv2-score.musicxml')
    l2 = ListeAccords(score2, instr = (K,decr,σ), classe = 'normal')
    l2.HarmonicDescriptors()

    space2 = ['harmonicity','concordanceTotale','concordanceOrdre3','tension']
    for descr in space2:
        listeDescr = l2.Liste(descr)
        maxDescr = max(listeDescr)
        listeDescr = [100*float(l)/maxDescr for l in listeDescr]
        Dic_Harm[(K,decr,σ)][descr] = {id: listeDescr[Dic_id[id]]  for id in Dic_id}

    # Enregistrement du dictionnaire des decripteurs harmoniques
    with open('Dic_Harm.pkl', 'wb') as f:
        pickle.dump(Dic_Harm, f)

    print('\n' + '-----------------------------' + '\nTimbre ({},{},{}) rajouté au dictionnaire'.format(K,decr,σ))

K_choice = [5,7,11,17]
decr_choice = [0,0.5,1]
σ_choice = [0.005,0.01]

# for K in K_choice:
#     for decr in decr_choice:
#         for σ in σ_choice:
#             AjoutTimbre(K,decr,σ)

for elt in Dic_Harm:
    print(elt)
    
# print(Dic_Harm[(11,1/2,0.005)]['concordance'])
