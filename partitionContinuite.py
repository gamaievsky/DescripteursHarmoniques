
'''
Etant donnée une liste d'accords, les trie par ordre de concordance, consonance, tension et concordance totale,
en affichant en dessous les valeurs. Prend en entrée dans le fichier paramètre deux listes de même taille : partiels,
qui contient l'emplacement des partiels (éventuellement inharmoniques),  et amplitudes, avec leurs amplitudes
respectives
'''

#from mimetypes import init

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from operator import itemgetter, attrgetter
from music21 import *
#from music21 import note, stream, corpus, tree, chord, pitch, converter

import parametres
import tunings


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


  def __init__(self, stream):

      self.stream = stream
      self.tree = tree.fromStream.asTimespans(stream, flatten=True,classList=(note.Note, chord.Chord))
      if parametres.mode=='manual':
          self.partiels = parametres.partiels
          self.amplitudes = parametres.amplitudes
      else:
          self.partiels = []
          self.amplitudes = []
          for i in range(parametres.K):
              self.partiels.append(i+1)
              self.amplitudes.append(1/(i+1)**parametres.decr)

      self.sig = parametres.sig
      self.temperament = tunings.Equal
                          #[0,0,0,0,0,0,0,0,0,0,0,0]#Tempere#
                          #[0,-10,+4,-6,+8,-2,-12,+2,-8,+6,-4,+10]#Pythagore
                          #[0,+17,-7,+10,-14,+3,+20,-3,+14,-10,+7,-17]#Mesotonique 1/4
                          #[]#Mesotonique 1/6
                          #[0,-29,+4,+16,-14,-2,-31,+2,-27,-16,-4,-12]#Juste Majeur
                          #[0,12,+4,+16,-13,-2,+32,+2,+14,-17,+18,-11]#Juste mineur

      self.noteDeReferencePourLeTunning = parametres.noteDeReferencePourLeTunning
      self.grandeursHarmoniques = []
      self.normalisation = [2,3,4,5,6,7,8]


      #self.ConcordanceCoherenceConcordanceOrdre3Liste()


  def  spectre(self,f0):

      '''Cette methode va etre appelee dans la classe Accord, mais elle est definie ici car le seul attribut d'objet
       qui en est parametre est l'instrument'''

      n = np.arange(0,16,0.001)
      S = np.zeros(np.shape(n))
      for i in range(1, len(self.partiels) + 1):
          S = S + (self.amplitudes[i-1]) * np.exp(-(n - np.log2(self.partiels[i-1] * f0))**2 / (2 * self.sig**2))
      return S

      #for i, elt in enumerate(self.instrument[0]):
    #      S = S + self.instrument[1][i] * np.exp(-(n - np.log2(elt * f0))**2 / (2 * (self.instrument[2][i])**2))
      #return S

  def Normalisation(self):

      """ Calcule la concordance d'ordre n de l'unisson a n notes, pour n allant de 2 a 8"""

      self.normalisation[0] = np.sum(self.spectre(100)*self.spectre(100))
      self.normalisation[1] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/3)
      self.normalisation[2] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/4)
      self.normalisation[3] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/5)
      self.normalisation[4] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/6)
      self.normalisation[5] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/7)
      self.normalisation[6] = (np.sum(self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)*self.spectre(100)))**(2/8)

  def frequenceAvecTemperament(self,pitch1):

      """Fonction qui prend en entree un pitch pour renvoyer une frequence, en tenant compte du temperament"""

      pitchRef = pitch.Pitch(self.noteDeReferencePourLeTunning)
      pitch1.microtone = self.temperament[(pitch1.pitchClass - pitchRef.pitchClass)%12] - 100*((pitch1.pitchClass - pitchRef.pitchClass)%12)

      return (pitch1.frequency)


  def ConcordanceCoherenceConcordanceOrdre3Liste (self):

      ''' Transforme chaque verticalite en objet Accord, calcule la concordance, la coherence et les concordances multiples, et stocke les resultats
      sous forme de liste d'Accords"
      '''

      self.Normalisation()
      spectreConcPrec = 0
      spectreConcTotPrec = 0
      nbrNotesPrec = 1
      baryConcPrec = 0
      baryConcTotPrec = 0

      for verticality in self.tree.iterateVerticalities():
          v = Accord(verticality, self.partiels, self.amplitudes, self.sig, self.normalisation, self.temperament,self.noteDeReferencePourLeTunning)
          if verticality.bassTimespan!=None :
              v.identifiant = verticality.bassTimespan.element.id

          v.ListeHauteursAvecMultiplicite()
          v.NombreDeNotes()

          if v.nombreDeNotes>=2:
              v.Dissonance()
              v.SpectreConcordance()
              v.Concordance()
              v.ProduitConc(spectreConcPrec)
              spectreConcPrec = v.spectreConcordance
              v.BaryConcordance()
              v.DifBaryConc(baryConcPrec)
              baryConcPrec = v.baryConc

              if v.nombreDeNotes>=3:
                  v.Tension()
                  v.SpectreConcordanceTot()
                  v.ConcordanceTotale()
                  v.ProduitConcTot(spectreConcTotPrec,nbrNotesPrec)
                  spectreConcTotPrec = v.spectreConcordanceTot
                  nbrNotesPrec = v.nombreDeNotes
                  v.BaryConcordanceTot()
                  v.DifBaryConcTot(baryConcTotPrec)
                  baryConcTotPrec = v.baryConcTot

                  #v.DifBaryConcTot(baryConc)

          self.grandeursHarmoniques.append(v)


  def getAnnotatedStream(self, resultList = ['Concordance']):
      r=0
      for gH in self.grandeursHarmoniques:

          if gH.verticality.bassTimespan != None :
              element = gH.verticality.bassTimespan.element
              if element.isNote or element.isChord:

                  dataString = ""

                  if 'Concordance' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(gH.concordance,2))
                  if 'ConcordanceOrdre3' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(10 * gH.concordanceOrdre3,2))
                  if 'ConcordanceTotale' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(10 * gH.concordanceTotale,2))
                  if 'Dissonance' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str (round(gH.dissonance,2))
                  if 'Tension' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str (round(gH.tension,2))
                  if 'ProduitConc' in resultList:
                      if dataString != '': dataString + " "
                      if type(gH.produitConc) != str:
                          dataString = dataString + "-" + str(round(100*gH.produitConc,2))
                  if 'ProduitConcTot' in resultList:
                      if dataString != '': dataString + " "
                      if type(gH.produitConcTot) != str:
                          dataString = dataString + "-" + str(round(100*gH.produitConcTot,2))
                  if 'BaryConc' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(gH.baryConc,2))
                  if 'BaryConcTot' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(gH.baryConcTot,2))
                  if 'DifBaryConc' in resultList:
                      if dataString != '': dataString + " "
                      if type(gH.difBaryConc) != str:
                          dataString = dataString + "-" + str(round(gH.difBaryConc,2))
                  if 'DifBaryConcTot' in resultList:
                      if dataString != '': dataString + " "
                      if type(gH.difBaryConcTot) != str:
                          dataString = dataString + "-" + str(round(gH.difBaryConcTot,2))

              if r == 0:
                  dataString = dataString  + "\n" + resultList[0]
                  r=1
              element.lyric = dataString




      #self.stream[0].addLyric('Voilà !')
      #print(self.stream[3][0])

      return tree.toStream.partwise(self.tree, self.stream)



  def Liste(self, axe = 'concordance'):
      liste = []
      for accord in self.grandeursHarmoniques:
          liste.append(getattr(accord, axe))
      return liste

  def Representation(self, axes =  ['concordance','dissonance','tension'], color = 'b', lien = True):
      if len(axes)==2 :
          l1 = self.Liste(axes[0])
          l2 = self.Liste(axes[1])
          fig, ax = plt.subplots()
          if lien: plt.plot(l1, l2, color+'--')
          plt.plot(l1, l2, color+'o')
          for i in range(len(l1)):
              ax.annotate(i+1, (l1[i], l2[i]))
          plt.xlabel(axes[0])
          plt.ylabel(axes[1])
          plt.title(title)
          plt.show()
      else:
          l1 = self.Liste(axes[0])
          l2 = self.Liste(axes[1])
          l3 = self.Liste(axes[2])
          fig = plt.figure(2)
          ax = fig.add_subplot(111, projection='3d')
          if lien: plt.plot(l1, l2, l3, color+'--')
          for i in range(len(l1)):
              ax.scatter(l1[i], l2[i], l3[i], c=color, marker='o')
              ax.text(l1[i], l2[i], l3[i], i+1, color='red')
          ax.set_xlabel(axes[0])
          ax.set_ylabel(axes[1])
          ax.set_zlabel(axes[2])
          ax.set_title('title')
          plt.show()








  def classementConc(self):
      s = stream.Measure()
      ts1 = meter.TimeSignature('C')
      s.insert(0, ts1)
      s.insert(0, clef.TrebleClef())

      self.stream.lyrics = {}
      self.getAnnotatedStream('concordance')
      self.grandeursHarmoniques.sort(key=attrgetter('concordance'), reverse=True)
      for gH in self.grandeursHarmoniques:
          element = gH.verticality.bassTimespan.element
          s.insert(-1, element)
      s[0].addLyric('Concordance')
      s.show()
      del s[0].lyrics[1]

  def classementConcTot(self):
      s2 = stream.Measure()
      ts1 = meter.TimeSignature('C')
      s2.insert(0, ts1)
      s2.insert(0, clef.TrebleClef())

      self.stream.lyrics = {}
      self.getAnnotatedStream(['concordanceTotale'])
      self.grandeursHarmoniques.sort(key=attrgetter('concordanceTotale'), reverse=True)
      for gH in self.grandeursHarmoniques:
          element = gH.verticality.bassTimespan.element
          s2.insert(-1, element)
      s2[0].addLyric('ConcTot')
      s2.show()
      del s2[0].lyrics[1]

  def classementDiss(self):
      s1 = stream.Measure()
      ts1 = meter.TimeSignature('C')
      s1.insert(0, ts1)
      s1.insert(0, clef.TrebleClef())

      self.stream.lyrics = {}
      self.getAnnotatedStream('dissonance')
      self.grandeursHarmoniques.sort(key=attrgetter('dissonance'), reverse=False)
      for gH in self.grandeursHarmoniques:
          element = gH.verticality.bassTimespan.element
          s1.insert(-1, element)
      s1[0].addLyric('Dissonance')
      s1.show()
      del s1[0].lyrics[1]

  def classementTens(self):
      s3 = stream.Measure()
      ts1 = meter.TimeSignature('C')
      s3.insert(0, ts1)
      s3.insert(0, clef.TrebleClef())

      self.stream.lyrics = {}
      self.getAnnotatedStream('tension')
      self.grandeursHarmoniques.sort(key=attrgetter('tension'), reverse=False)
      for gH in self.grandeursHarmoniques:
          element = gH.verticality.bassTimespan.element
          s3.insert(-1, element)
      s3[0].addLyric('Tension')
      s3.show()
      del s3[0].lyrics[1]






class Accord(ListeAccords):
  '''
  Classe qui traite les verticalites en heritant de l'instrument et de la methode spectre de la classe ListeAccords,
  et ayant comme attributs supplementaires les grandeurs lies a la concordance
  Faiblesse pour l'instant : l'arbre de la classe mere est vide, un attribut 'verticality' vient le remplacer
  '''


  def __init__(self, verticality, partiels, amplitudes, sig, normalisation, temperament, noteDeReferencePourLeTunning):# verticality

      self.partiels = partiels
      self.amplitudes = amplitudes
      self.sig = sig
      self.temperament = temperament
      self.noteDeReferencePourLeTunning = noteDeReferencePourLeTunning
      self.normalisation = normalisation
      self.listeHauteursAvecMultiplicite = []
      self.verticality = verticality
      self.concordance = 0
      self.concordanceTotale = 0
      self.concordanceOrdre3 = 0
      self.dissonance = 0
      self.tension = 0
      self.identifiant = 0
      self.nombreDeNotes = 0
      self.spectreConcordance = 0
      self.spectreConcordanceTot = 0
      self.produitConc = 0
      self.produitConcTot = 0
      self.baryConc= 0
      self.baryConcTot = 0
      self.difBaryConc = 0
      self.difBaryConcTot = 0




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

  def Dissonance(self):

       for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
           for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
               if (i<j):
                   for k1 in range(1,len(self.partiels) + 1):
                       for k2 in range(1,len(self.partiels) + 1):
                           fmin = min(self.partiels[k2-1] * self.frequenceAvecTemperament(pitch2), self.partiels[k1-1] * self.frequenceAvecTemperament(pitch1))
                           fmax = max(self.partiels[k2-1] * self.frequenceAvecTemperament(pitch2), self.partiels[k1-1] * self.frequenceAvecTemperament(pitch1))
                           s=0.24/(0.021*fmin+19.)
                           self.dissonance = self.dissonance + (100 * self.amplitudes[k1-1] * self.amplitudes[k2-1]) * (np.exp(-3.5*s*(fmax-fmin))-np.exp(-5.75*s*(fmax-fmin)))
       n = self.nombreDeNotes
       self.dissonance = self.dissonance/(self.nombreDeNotes*(self.nombreDeNotes - 1)/2)


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
      #self.concordanceOrdre3 = np.log2(1 + self.concordanceOrdre3 / (self.normalisation[1]*(self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)**(2/3)))
      #self.concordanceOrdre3 = np.log2(1 + self.concordanceOrdre3)/(np.log(1 + self.normalisation[1] * (self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)**(2/3)) / np.log(1 + self.normalisation[1]))

  def SpectreConcordance(self):
      self.spectreConcordance = np.zeros(16000)
      for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
          for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
              if (i<j):
                  self.spectreConcordance = self.spectreConcordance + self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))


  def Concordance(self):

      """ Normalisation logarithmique, de maniere a rendre egales les concordances des unissons de n notes"""
      self.concordance = np.sum(self.spectreConcordance)
      self.concordance = self.concordance/(self.nombreDeNotes*(self.nombreDeNotes - 1)/2)

      #self.concordance = np.log2(1 + self.concordance / (self.normalisation[0]*self.nombreDeNotes*(self.nombreDeNotes - 1)/2))
      #self.concordance = np.log2(1 + self.concordance)/(np.log(1 + self.normalisation[0]*self.nombreDeNotes*(self.nombreDeNotes - 1)/2) / np.log(1 + self.normalisation[0]))


  def ProduitConc(self, spectreConcPrec):
      if type(spectreConcPrec) is int:
          self.produitConc = ""
      elif parametres.norm1: # Version normalisée par rapport à la première concordance
          self.produitConc = (np.sum(self.spectreConcordance * spectreConcPrec) / np.sum(spectreConcPrec * spectreConcPrec))
      else: # Version normalisée
          self.produitConc = np.sum(self.spectreConcordance * spectreConcPrec)**(1/2)


  def BaryConcordance(self):
      self.baryConc = np.sum(self.spectreConcordance * np.arange(0,16,0.001)) / (self.concordance * (self.nombreDeNotes*(self.nombreDeNotes - 1)/2))


  def DifBaryConc(self,baryConcPrec): # Calcule les variations de barycentre
      if baryConcPrec == 0:
          self.difBaryConc = ""
      else:
          self.difBaryConc = self.baryConc - baryConcPrec
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

  def ProduitConcTot(self, spectreConcTotPrec, nbrNotesPrec):
      if type(spectreConcTotPrec) is int:
          self.produitConcTot = ""
      elif parametres.norm2: # Version normalisée par rapport à la première concordance
          self.produitConcTot = (np.sum(self.spectreConcordanceTot * spectreConcTotPrec) / np.sum(spectreConcTotPrec * spectreConcTotPrec))**(2/self.nombreDeNotes)
      else: #Version non normalisée
          self.produitConcTot = np.sum(self.spectreConcordanceTot * spectreConcTotPrec)**(2/(self.nombreDeNotes * nbrNotesPrec))


  def BaryConcordanceTot(self):
      self.baryConcTot = np.sum(self.spectreConcordanceTot * np.arange(0,16,0.001)) / (self.concordanceTotale ** (self.nombreDeNotes/2))



  def DifBaryConcTot(self,baryConcTotPrec): # Calcule les variations de barycentre
      if baryConcTotPrec == 0:
          self.difBaryConcTot = ""
      else:
          self.difBaryConcTot = self.baryConcTot - baryConcTotPrec



title = 'SuiteAccords'
# Palestrina, AccordsMajeurs, AccordsMineur, Majeur3et4notes, Majeur3et4notes, Accords3Notes, DispoMajeurMineur, Tension
# Cadence3V, Cadence4VMaj, Cadence4Vmin, SuiteAccords
score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/'+title+'.musicxml')

l = ListeAccords(score)
l.ConcordanceCoherenceConcordanceOrdre3Liste()
# Afficher : Concordance, Dissonance, Tension, ConcordanceOrdre3, ConcordanceTotale, ProduitConc, ProduitConcTot,
#            BaryConc, BaryConcTot, DifBaryConc, DifBaryConcTot
l.getAnnotatedStream(['Dissonance'])
l.stream.show()
#l.Representation(['concordance','dissonance'], 'r', False)
