
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
              v.Dissonance()
              v.SpectreConcordance()
              v.Concordance()
              v.BaryConcordance()
              v.CrossConcordance(Prec)
              v.HarmonicChange(Prec)
              v.DiffConcordance(Prec)
              v.DifBaryConc(Prec)

              if v.nombreDeNotes>=3:
                  v.Tension()
                  v.SpectreConcordanceTot()
                  v.ConcordanceTotale()
                  v.CrossConcordanceTot(Prec)
                  v.BaryConcordanceTot()
                  v.DifBaryConcTot(Prec)

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
          liste.append(getattr(accord, axe))
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


  def AbstractRepresentation(self, space =  ['concordance','dissonance','tension']):
      if len(space)==2 :
          color = parametres.color_abstr
          l1 = self.Liste(space[0])
          l2 = self.Liste(space[1])
          fig, ax = plt.subplots()
          if parametres.link: plt.plot(l1, l2, color+'--')
          plt.plot(l1, l2, color+'o')
          for i in range(len(l1)):
              ax.annotate(i+1, (l1[i], l2[i]))
          plt.xlabel(space[0])
          plt.ylabel(space[1])
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


  def __init__(self, verticality, partiels, amplitudes, sig, energyUniss, temperament, noteDeReferencePourLeTunning):# verticality

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
      self.dissonance = 0
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
      self.harmonicChange = 0
      self.diffConcordance = 0




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
                           s = 0.44*(np.log(parametres.β2/parametres.β1)/(parametres.β2-parametres.β1))*(fmax-fmin)/(fmin**(0.477))
                           diss = np.exp(-parametres.β1*s)-np.exp(-parametres.β2*s)
                           if parametres.type_diss=='produit':
                               self.dissonance = self.dissonance + (self.amplitudes[k1-1] * self.amplitudes[k2-1]) * diss
                           elif parametres.type_diss == 'minimum':
                               self.dissonance = self.dissonance + min(self.amplitudes[k1-1],self.amplitudes[k2-1]) * diss

       if parametres.norm_diss:
           if parametres.type_diss=='produit':
               self.dissonance = self.dissonance/self.energyUniss[0]
           elif parametres.type_diss == 'minimum':
               self.dissonance = self.dissonance/np.sqrt(self.energyUniss[0])

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
                      self.diffConcordance = self.diffConcordance + np.divide(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))), np.sqrt(self.energy * Prec.energy))
                  if parametres.norm_diffConc=='energy + conc':
                      self.diffConcordance = self.diffConcordance + np.divide(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))), np.sqrt(self.energy * Prec.energy)* self.concordance*Prec.concordance)
                  if parametres.norm_diffConc=='first':
                      self.diffConcordance = self.diffConcordance + np.divide(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))), np.sqrt(self.energy * Prec.energy)* Prec.concordance)




title = 'SuiteAccordsPiano'
# Palestrina, AccordsMajeurs, AccordsMineur, Majeur3et4notes, Majeur3et4notes, Accords3Notes, DispoMajeurMineur, Tension
# Cadence3V, Cadence4VMaj, Cadence4Vmin, SuiteAccords
score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/ExemplesMusicaux/'+title+'.musicxml')
#score = converter.parse('/Users/manuel/Github/DescripteursHarmoniques/Exemples/AccordsMineur.musicxml')
l = ListeAccords(score)
l.HarmonicDescriptors()
# Afficher : Concordance, Dissonance, Tension, ConcordanceOrdre3, ConcordanceTotale, crossConcordance, crossConcordanceTot,
#            BaryConc, BaryConcTot, DifBaryConc, DifBaryConcTot
space = ['tension','diffConcordance','crossConcordanceTot','harmonicChange']
# l.Classement('concordance')
l.getAnnotatedStream(['concordance'])
l.stream.show()
# l.Affichage(space)
