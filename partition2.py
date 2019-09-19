'''
Etant donnée une partition, affiche en-dessous les valeurs de concordance, dissonance, tension, concordance totale
'''


#from mimetypes import init
import numpy as np
from operator import itemgetter, attrgetter
from music21 import *
#from music21 import note, stream, corpus, tree, chord, pitch, converter

K = 15
decr = 1
sig = 0.01


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
      self.K = parametres.K
      self.decr = parametres.decr
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
      for i in range(1, self.K + 1):
          S = S + (1/i ** self.decr) * np.exp(-(n - np.log2(i * f0))**2 / (2 * self.sig**2))
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

      for verticality in self.tree.iterateVerticalities():
          v = Accord(verticality, self.K, self.decr, self.sig, self.normalisation, self.temperament,self.noteDeReferencePourLeTunning)
          if verticality.bassTimespan!=None :
              v.identifiant = verticality.bassTimespan.element.id

          v.ListeHauteursAvecMultiplicite()
          v.ListeConcordanceDesIntervallesDansAccord()
          v.NombreDeNotes()
          if v.nombreDeNotes>=2:
              v.Concordance()
              v.Dissonance()
              if v.nombreDeNotes>=3:
                  v.Tension()
                  v.Coherence()
                  v.ConcordanceOrdre3()
                  if v.nombreDeNotes>=4:
                      v.ConcordanceTotale()
                  else:
                      v.concordanceTotale = v.concordanceOrdre3
              else:
                  v.concordanceTotale = v.concordance


          self.grandeursHarmoniques.append(v)


  def getAnnotatedStream(self, resultList = ['concordance']):

      for gH in self.grandeursHarmoniques:

          if gH.verticality.bassTimespan != None :
              element = gH.verticality.bassTimespan.element
              if element.isNote or element.isChord:

                  dataString = ""

                  if 'concordance' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(gH.concordance,2))
                  if 'concordanceOrdre3' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str(round(10 * gH.concordanceOrdre3,2))
                  if 'coherence' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str (round(gH.coherence,2))
                  if 'dissonance' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str (round(gH.dissonance,2))
                  if 'tension' in resultList:
                      if dataString != '': dataString + " "
                      dataString = dataString + str (round(gH.tension,2))

              element.lyric = dataString




      return tree.toStream.partwise(self.tree, self.stream)

  def moyenneConcordance (self):
      l = []
      for accord in self.grandeursHarmoniques:
          l.append(accord.concordance)
      return np.mean(l)

  def moyenneCoherence (self):
      l = []
      for accord in self.grandeursHarmoniques:
          l.append(accord.coherence)
      return np.mean(l)

  def moyenneConcordanceTotale (self):
      l = []
      for accord in self.grandeursHarmoniques:
          l.append(accord.concordanceTotale)
      return np.mean(l)

  def moyenneConcordanceOrdre3 (self):
      l = []
      for accord in self.grandeursHarmoniques:
          l.append(accord.concordanceOrdre3)
      return np.mean(l)


  def offsetList (self):

      '''Donne la liste de tous les offsets des verticalites'''

      l = []
      for verticality in self.tree.iterateVerticalities():
          v = Accord(verticality)
          l.append(v.offset)
      return l

  def idList (self):

      '''Donne la liste des identifiants des verticalites'''

      l = []
      for verticality in self.tree.iterateVerticalities():
          v = Accord(verticality)
          l.append(v.id)
      return l

  #def IDToTimeList (self, ID_list):
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


  def classementConcTot(self):
      s2 = stream.Measure()
      ts1 = meter.TimeSignature('C')
      s2.insert(0, ts1)
      s2.insert(0, clef.TrebleClef())

      self.stream.lyrics = {}
      self.getAnnotatedStream(['concordanceOrdre3'])
      self.grandeursHarmoniques.sort(key=attrgetter('concordanceOrdre3'), reverse=True)
      for gH in self.grandeursHarmoniques:
          element = gH.verticality.bassTimespan.element
          s2.insert(-1, element)
      s2[0].addLyric('ConcTot')
      s2.show()
      del s2[0].lyrics[1]

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

        #sorted(self.grandeursHarmoniques, key=attrgetter('concordance'))

      #return tree.toStream.partwise(self.tree, self.stream)





class Accord(ListeAccords):
  '''
  Classe qui traite les verticalites en heritant de l'instrument et de la methode spectre de la classe ListeAccords,
  et ayant comme attributs supplementaires les grandeurs lies a la concordance
  Faiblesse pour l'instant : l'arbre de la classe mere est vide, un attribut 'verticality' vient le remplacer
  '''


  def __init__(self, verticality, K, decr, sig, normalisation, temperament, noteDeReferencePourLeTunning):# verticality

      self.K = K
      self.decr = decr
      self.sig = sig
      self.temperament = temperament
      self.noteDeReferencePourLeTunning = noteDeReferencePourLeTunning
      self.normalisation = normalisation
      self.listeHauteursAvecMultiplicite = []
      self.listeConcordanceDesIntervallesDansAccord = []
      self.verticality = verticality
      self.concordance = 0
      self.coherence = 0
      self.concordanceTotale = 0
      self.concordanceOrdre3 = 0
      self.dissonance = 0
      self.tension = 0
      self.identifiant = 0
      self.nombreDeNotes = 0


  def __repr__(self):
      """Affichage"""
      return "Concordance: {0} \nCoherence: {1}\nConcordance d'ordre 3:  {2} \nConcordance totale: {3}".format(self.concordance, self.coherence,self.concordanceOrdre3,self.concordanceTotale)



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


  def ListeConcordanceDesIntervallesDansAccord(self):

      '''Cree la liste des concordances des intervalles qui constituent l'accord, et le fixe comme parametre, ceci afin d'eviter
      les redondances dans les calculs de la concordance et de la coherence '''

      for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
          for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
              if (i<j):
                  self.listeConcordanceDesIntervallesDansAccord.append(np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))))

  def NombreDeNotes(self):
      if self.listeHauteursAvecMultiplicite != None:
          self.nombreDeNotes = len(self.listeHauteursAvecMultiplicite)


  def Concordance(self):

      """ Normalisation logarithmique, de maniere a rendre egales les concordances des unissons de n notes"""

      self.concordance = np.sum(self.listeConcordanceDesIntervallesDansAccord)
      #self.concordance = np.log2(1 + self.concordance / (self.normalisation[0]*self.nombreDeNotes*(self.nombreDeNotes - 1)/2))
      #self.concordance = np.log2(1 + self.concordance)/(np.log(1 + self.normalisation[0]*self.nombreDeNotes*(self.nombreDeNotes - 1)/2) / np.log(1 + self.normalisation[0]))




  def Coherence(self):

      self.coherence = np.std(self.listeConcordanceDesIntervallesDansAccord)
      if self.nombreDeNotes == 3 :
          self.coherence = self.coherence/self.normalisation[1]
      elif self.nombreDeNotes == 4 :
          self.coherence = self.coherence/self.normalisation[2]
      elif self.nombreDeNotes == 5 :
          self.coherence = self.coherence/self.normalisation[3]
      elif self.nombreDeNotes == 6 :
          self.coherence = self.coherence/self.normalisation[4]
      elif self.nombreDeNotes == 7 :
          self.coherence = self.coherence/self.normalisation[5]
      elif self.nombreDeNotes == 8 :
          self.coherence = self.coherence/self.normalisation[6]

  def ConcordanceTotale(self):

        S = np.ones(16000)
        for pitch in self.listeHauteursAvecMultiplicite:
                S = S*self.spectre(self.frequenceAvecTemperament(pitch))
                self.concordanceTotale = np.sum(S)

        self.concordanceTotale = self.concordanceTotale**(2/self.nombreDeNotes)

        if self.nombreDeNotes == 4 :
            self.concordanceTotale = np.log2(1 + self.concordanceTotale/self.normalisation[2])
        elif self.nombreDeNotes == 5 :
            self.concordanceTotale = np.log2(1 + self.concordanceTotale/self.normalisation[3])
        elif self.nombreDeNotes == 6 :
            self.concordanceTotale = np.log2(1 + self.concordanceTotale/self.normalisation[4])
        elif self.nombreDeNotes == 7 :
            self.concordanceTotale = np.log2(1 + self.concordanceTotale/self.normalisation[5])
        elif self.nombreDeNotes == 8 :
            self.concordanceTotale = np.log2(1 + self.concordanceTotale/self.normalisation[6])


  def Dissonance(self):

      for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
          for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
              if (i<j):
                  for k1 in range(1,self.K + 1):
                      for k2 in range(1,self.K + 1):
                          x = abs(np.log2((k2*self.frequenceAvecTemperament(pitch2)) /  (k1*self.frequenceAvecTemperament(pitch1))))
                          b1 = 0.8
                          b2 = 1.6
                          b3 = 4.0
                          self.dissonance = self.dissonance + (b3/(k1*k2) ** self.decr) * (np.exp(-12*b1*x) - np.exp(-12*b2*x))


  def Tension(self):

      for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
          for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
              for l, pitch3 in enumerate(self.listeHauteursAvecMultiplicite):
                  if (i<j<l):
                      for k1 in range(1,self.K + 1):
                          for k2 in range(1,self.K + 1):
                              for k3 in range(1,self.K + 1):
                                  x = np.log2((k2*self.frequenceAvecTemperament(pitch2)) / (k1*self.frequenceAvecTemperament(pitch1)))
                                  y = np.log2((k3*self.frequenceAvecTemperament(pitch3)) / (k2*self.frequenceAvecTemperament(pitch2)))
                                  z = x + y
                                  X = abs(x)
                                  Y = abs(y)
                                  Z = abs(z)
                                  a = 0.6
                                  self.tension = self.tension = self.tension + (1/(k1*k2*k3) ** self.decr) * max(np.exp(-(12*(X-Y)/a)**2) , np.exp(-(12*(Y-Z)/a)**2) , np.exp(-(12*(X-Z)/a)**2))



  def ConcordanceOrdre3(self):

      for i, pitch1 in enumerate(self.listeHauteursAvecMultiplicite):
          for j, pitch2 in enumerate(self.listeHauteursAvecMultiplicite):
              for k, pitch3 in enumerate(self.listeHauteursAvecMultiplicite):
                  if (i<j<k):
                      self.concordanceOrdre3 =  self.concordanceOrdre3 + np.sum(self.spectre(self.frequenceAvecTemperament(pitch1))*self.spectre(self.frequenceAvecTemperament(pitch2))*self.spectre(self.frequenceAvecTemperament(pitch3)))

      self.concordanceOrdre3 = self.concordanceOrdre3**(2/3)
      #self.concordanceOrdre3 = np.log2(1 + self.concordanceOrdre3 / (self.normalisation[1]*(self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)**(2/3)))
      #self.concordanceOrdre3 = np.log2(1 + self.concordanceOrdre3)/(np.log(1 + self.normalisation[1] * (self.nombreDeNotes*(self.nombreDeNotes - 1)*(self.nombreDeNotes - 2)/6)**(2/3)) / np.log(1 + self.normalisation[1]))




score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Disposition/AccordsMajeurs.musicxml')
#score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Disposition/AccordsMineur.musicxml')
#score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Disposition/Tension.musicxml')
#score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Disposition/DispoMajeurMineur.musicxml')
l = ListeAccords(score)
l.ConcordanceCoherenceConcordanceOrdre3Liste()
l.classementConc()
l.classementDiss()
l.classementConcTot()
l.classementTens()
