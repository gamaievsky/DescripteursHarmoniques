import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from p5 import *
import p5
from music21 import *
from music21.converter.subConverters import ConverterMusicXML
from Tonnetz_classes import ListeAccords, Accord
from tkinter import *
import tkinter as tk
from tkinter.ttk import *
import copy
import parametres
import pickle
from PIL import Image
import os

# Géométrie
geom = (3,4,7)

largeur = 700
hauteur = 700

# Valeurs par défaut
plot_dim, plot_aug, plot_quarte = True, True, True
pitch_liste = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
timbre = parametres.timbre_def
K, decr, σ = timbre[0], timbre[1], timbre[2]
L_mem, decr_mem = parametres.memory_size, parametres.memory_decr_ponderation
colorR, colorG, colorB, colorJet  = 'diffRoughness','harmonicChange','diffConcordance','defaut'
conv_musicxml = ConverterMusicXML()

with open('numSave_Ton.pkl', 'rb') as f:
    numSave_Ton = pickle.load(f)

# Couleurs
class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)
COL = MplColorHelper('jet', 0, 100)



def setup():
    size(largeur, hauteur)
    title('Tonnetz [{},{},{}]'.format(geom[0],geom[1],geom[2]))
    color_mode('RGBA', 100)

    global image_score
    image_score = None

    global stream1, tot_harmonicChange, tot_diffConcordance, tot_diffRoughness
    stream1 = stream.Stream()
    tot_harmonicChange, tot_diffConcordance, tot_diffRoughness = [], [], []

    global grid
    grid = Grid('Bb', x = width/2, y = height - 100, nb_col=5, hauteur=6)
    grid.Fill_coord()
    grid.Fill_chords()
    grid.Fill_chords2()

    global f,f0
    f = create_font("Arial.ttf", 16)
    f0 = create_font("Arial.ttf", 14)

def draw():
    background(255)
    grid.draw()
    grid.show()

    # Partition
    if image_score != None:
        α = image_score.width / image_score.height
        h = 50
        image(image_score, (200, 10), (α*h,h))

    # Légende des descripteurs totalisés
    if len(tot_harmonicChange)>=2:
        text_font(f0)
        fill(0)
        text_align("CENTER")
        p5.text('Somme sur les transitions : HarmonicChange : {}, DiffConcordance = {}, DiffRoughness : {}'.format(int(np.sum(tot_harmonicChange[1:])), int(np.sum(tot_diffConcordance[1:])), int(np.sum(tot_diffRoughness[1:]))),(largeur/2,hauteur-70))

    # Légende
    text_font(f0)
    fill(50)
    text_align("CENTER")
    p5.text('Tonnetz [{},{},{}]'.format(geom[0],geom[1],geom[2]) + ', timbre (K: {}, decr: {}, σ: {})'.format(K, decr, σ) + ', mémoire ({} acc, decr: {})'.format(L_mem + 1, decr_mem),(largeur/2,hauteur-30))

def mouse_pressed():
    global numSave_Ton

    # Bouton Paramètres
    if (0 < mouse_x < 160) and (0 < mouse_y < 25):
        no_loop()
        window.deiconify()
        window.mainloop()
        loop()
    # Bouton Save
    if (largeur-90 < mouse_x < 1200) and (0 < mouse_y < 25):
        stroke(250)
        fill(250)
        rect((largeur-90, 0), 90, 25)
        fill(250)
        rect((0, 0), 160, 25)
        save(filename='/Users/manuel/Dropbox (TMG)/Thèse/Tonnetz/tonnetz{}.png'.format(numSave_Ton))
        print('file {} saved'.format(numSave_Ton))
        numSave_Ton += 1
        with open('numSave_Ton.pkl', 'wb') as f:
            pickle.dump(numSave_Ton, f)

    grid.click()







class Pitch:
    def __init__(self, pitch,x,y,R):
        self.pitch = pitch
        self.x = x
        self.y = y
        self.R = R

    def draw(self):
        stroke(0)
        fill(255)
        circle((self.x,self.y), self.R*2, mode = 'CENTER')

        text_font(f)
        fill(0)
        text_align("CENTER")
        # text(self.pitch, (self.x, self.y - p5.text_ascent()/2))
        p5.text(self.pitch, (self.x, self.y - text_ascent()/2))


class Chord:
    def __init__(self, pitch_bass, type, x, y, d, R):
        self.pitch_bass = pitch_bass
        self.type = type
        self.x = x
        self.y = y
        self.d = d
        self.R = R
        self.pitches = [Pitch(pitch_bass, x, y, self.R)]
        xt, yt= x + (-1)**(type == 'minor')*d*np.sqrt(3)/2, y-d/2
        self.pitches.append(Pitch(pitch_liste[(pitch_liste.index(pitch_bass) + geom[1]*(self.type=='major') + geom[0]*(self.type=='minor')) % 12], xt, yt, self.R))
        self.pitches.append(Pitch(pitch_liste[(pitch_liste.index(pitch_bass) + geom[2]) % 12], x, y-d, self.R))
        self.stream = stream.Stream()
        self.harmonicChange = 100
        self.diffConcordance = 100
        self.diffRoughness = 100
        self.selec = False
        self.dic_color = {}

    def draw(self):
        var = [colorR,colorG,colorB,colorJet]
        for i,col in enumerate(['R','G','B','Jet']):
            if var[i] == 'harmonicChange':
                self.dic_color[col]= self.harmonicChange
            elif var[i] == 'diffConcordance':
                self.dic_color[col] = self.diffConcordance
            elif var[i] == 'diffRoughness':
                self.dic_color[col] = self.diffRoughness
            else: self.dic_color[col] = 0

        if colorJet == 'defaut':
            fill(self.dic_color['R'], self.dic_color['G'], self.dic_color['B'])
        else:
            z = self.dic_color['Jet']
            fill(100*COL.get_rgb(z)[0],100*COL.get_rgb(z)[1],100*COL.get_rgb(z)[2])

        stroke(0)
        stroke_weight(1 + 2*self.selec)
        xt, yt= self.x + (-1)**(self.type == 'minor')*self.d*np.sqrt(3)/2, self.y-self.d/2
        triangle((self.x,self.y), (xt,yt), (self.x,self.y-self.d))
        self.pitches[0].draw()
        self.pitches[1].draw()
        self.pitches[2].draw()

        α1, β1 = (self.y-yt)/(self.x-xt), (self.x*yt - self.y*xt)/(self.x-xt)
        α2, β2 = (yt - (self.y-self.d))/(xt-self.x), (xt*(self.y-self.d) - yt*self.x)/(xt-self.x)


    def show(self):
        global mouse_found
        if not mouse_found:
            xt, yt= self.x + (-1)**(self.type == 'minor')*self.d*np.sqrt(3)/2, self.y-self.d/2
            α1, β1 = (self.y-yt)/(self.x-xt), (self.x*yt - self.y*xt)/(self.x-xt)
            α2, β2 = (yt - (self.y-self.d))/(xt-self.x), (xt*(self.y-self.d) - yt*self.x)/(xt-self.x)
            if ((mouse_x > self.x and self.type=='major') or (mouse_x < self.x and self.type=='minor')) and (mouse_y < α1*mouse_x + β1) and (mouse_y > α2*mouse_x + β2):
                print('\nAccord {} [{} - {} - {}]'.format(self.type, self.pitches[0].pitch, self.pitches[1].pitch, self.pitches[2].pitch) + '\nHarmonicChange: {}\nDiffConcordance: {}\nDiffRoughness: {}'.format(int(self.harmonicChange), int(self.diffConcordance), int(self.diffRoughness)))
                mouse_found = True


    def click(self):
        global stream1, image_score, tot_harmonicChange, tot_diffConcordance, tot_diffRoughness
        xt, yt= self.x + (-1)**(self.type == 'minor')*self.d*np.sqrt(3)/2, self.y-self.d/2
        α1, β1 = (self.y-yt)/(self.x-xt), (self.x*yt - self.y*xt)/(self.x-xt)
        α2, β2 = (yt - (self.y-self.d))/(xt-self.x), (xt*(self.y-self.d) - yt*self.x)/(xt-self.x)
        if ((mouse_x > self.x and self.type=='major') or (mouse_x < self.x and self.type=='minor')) and (mouse_y < α1*mouse_x + β1) and (mouse_y > α2*mouse_x + β2):
            # print('\nAccord {} - {} - {}'.format(self.pitches[0].pitch, self.pitches[1].pitch, self.pitches[2].pitch) + '\nHarmonicChange: {}\nDiffConcordance: {}\nDiffRoughness: {}'.format(self.harmonicChange, self.diffConcordance, self.diffRoughness))
            print('\n................')
            print('Nouvel accord {} : [{}-{}-{}]'.format(self.type, self.pitches[0].pitch, self.pitches[1].pitch, self.pitches[2].pitch))
            tot_harmonicChange.append(self.harmonicChange)
            tot_diffConcordance.append(self.diffConcordance)
            tot_diffRoughness.append(self.diffRoughness)


            def mod(s):
                if s[-1] == 'b': s = s[:-1] + '-'
                s += '4'
                return s

            c = chord.Chord([mod(self.pitches[0].pitch), mod(self.pitches[1].pitch), mod(self.pitches[2].pitch)])
            stream1.append(c)

            # Partition
            global image_score
            title = 'accord'
            filepath = '/Users/manuel/Dropbox (TMG)/Thèse/Tonnetz/Partition/'
            out_filepath = conv_musicxml.write(stream1, 'musicxml', fp=filepath + title + '.XML', subformats=['png'])
            # im = Image.open(out_filepath)
            # width, height = im.size
            # region = im.crop((left, 0, right, height))
            # region.save(out_filepath[:-6] + '.png')
            os.remove(filepath + title + '.XML')
            image_score = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Tonnetz/Partition/accord-1.png')

            self.selec = True
            return True
        else: return False


class Chord2:
    def __init__(self, pitch_center, type, x, y, R, L):
        self.pitch_center = pitch_center
        self.type = type
        self.x = x
        self.y = y
        self.R = R
        self.L = L
        self.pitches = [pitch_liste[(pitch_liste.index(pitch_center) - (type=='dim')*geom[0] - (type=='aug')*geom[1] - (type=='quarte')*geom[2]) % 12], pitch_center ,pitch_liste[(pitch_liste.index(pitch_center) + (type=='dim')*geom[0] + (type=='aug')*geom[1] + (type=='quarte')*geom[2]) % 12]]
        self.stream = stream.Stream()
        self.harmonicChange = 100
        self.diffConcordance = 100
        self.diffRoughness = 100
        self.selec = False
        self.dic_color = {}

    def draw(self):
        var = [colorR,colorG,colorB,colorJet]
        for i,col in enumerate(['R','G','B','Jet']):
            if var[i] == 'harmonicChange':
                self.dic_color[col]= self.harmonicChange
            elif var[i] == 'diffConcordance':
                self.dic_color[col] = self.diffConcordance
            elif var[i] == 'diffRoughness':
                self.dic_color[col] = self.diffRoughness
            else: self.dic_color[col] = 0

        if colorJet == 'defaut':
            fill(self.dic_color['R'], self.dic_color['G'], self.dic_color['B'])
        else:
            z = self.dic_color['Jet']
            fill(100*COL.get_rgb(z)[0],100*COL.get_rgb(z)[1],100*COL.get_rgb(z)[2])

        stroke(0)

        if self.type == 'dim':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x + self.R, self.y), (self.x + self.R + self.L*np.sqrt(3)/2,self.y + self.L/2), (self.x + self.R/2 + self.L*np.sqrt(3)/2,self.y + self.R*np.sqrt(3)/2 + self.L/2)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x - self.R, self.y), (self.x - self.R - self.L*np.sqrt(3)/2,self.y - self.L/2), (self.x - self.R/2 - self.L*np.sqrt(3)/2,self.y - self.R*np.sqrt(3)/2 - self.L/2)
            quad((x1,y1),(x2,y2),(x3,y3),(x4,y4))
            quad((X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4))
        if self.type == 'aug':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R, self.y), (self.x - self.R - self.L*np.sqrt(3)/2,self.y + self.L/2), (self.x - self.R/2 - self.L*np.sqrt(3)/2,self.y + self.R*np.sqrt(3)/2 + self.L/2)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R, self.y), (self.x + self.R + self.L*np.sqrt(3)/2,self.y - self.L/2), (self.x + self.R/2 + self.L*np.sqrt(3)/2,self.y - self.R*np.sqrt(3)/2 - self.L/2)
            quad((x1,y1),(x2,y2),(x3,y3),(x4,y4))
            quad((X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4))
        if self.type == 'quarte':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2 + self.L), (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2 + self.L)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2 - self.L), (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2 - self.L)
            quad((x1,y1),(x2,y2),(x3,y3),(x4,y4))
            quad((X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4))

    def show(self):
        global mouse_found
        def coeffDr(x1,y1,x2,y2):
            α = (y1-y2)/(x1-x2)
            β = (x1*y2 - x2*y1)/(x1-x2)
            return α, β

        if self.type == 'dim':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x + self.R, self.y), (self.x + self.R + self.L*np.sqrt(3)/2,self.y + self.L/2), (self.x + self.R/2 + self.L*np.sqrt(3)/2,self.y + self.R*np.sqrt(3)/2 + self.L/2)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x - self.R, self.y), (self.x - self.R - self.L*np.sqrt(3)/2,self.y - self.L/2), (self.x - self.R/2 - self.L*np.sqrt(3)/2,self.y - self.R*np.sqrt(3)/2 - self.L/2)
            cond1 = (mouse_y > coeffDr(x1,y1,x2,y2)[0]*mouse_x + coeffDr(x1,y1,x2,y2)[1]) and (mouse_y > coeffDr(x2,y2,x3,y3)[0]*mouse_x + coeffDr(x2,y2,x3,y3)[1]) and (mouse_y < coeffDr(x3,y3,x4,y4)[0]*mouse_x + coeffDr(x3,y3,x4,y4)[1]) and (mouse_y < coeffDr(x1,y1,x4,y4)[0]*mouse_x + coeffDr(x1,y1,x4,y4)[1])
            cond2 = (mouse_y < coeffDr(X1,Y1,X2,Y2)[0]*mouse_x + coeffDr(X1,Y1,X2,Y2)[1]) and (mouse_y < coeffDr(X2,Y2,X3,Y3)[0]*mouse_x + coeffDr(X2,Y2,X3,Y3)[1]) and (mouse_y > coeffDr(X3,Y3,X4,Y4)[0]*mouse_x + coeffDr(X3,Y3,X4,Y4)[1]) and (mouse_y > coeffDr(X1,Y1,X4,Y4)[0]*mouse_x + coeffDr(X1,Y1,X4,Y4)[1])
        if self.type == 'aug':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R, self.y), (self.x - self.R - self.L*np.sqrt(3)/2,self.y + self.L/2), (self.x - self.R/2 - self.L*np.sqrt(3)/2,self.y + self.R*np.sqrt(3)/2 + self.L/2)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R, self.y), (self.x + self.R + self.L*np.sqrt(3)/2,self.y - self.L/2), (self.x + self.R/2 + self.L*np.sqrt(3)/2,self.y - self.R*np.sqrt(3)/2 - self.L/2)
            cond1 = (mouse_y > coeffDr(x1,y1,x2,y2)[0]*mouse_x + coeffDr(x1,y1,x2,y2)[1]) and (mouse_y > coeffDr(x2,y2,x3,y3)[0]*mouse_x + coeffDr(x2,y2,x3,y3)[1]) and (mouse_y < coeffDr(x3,y3,x4,y4)[0]*mouse_x + coeffDr(x3,y3,x4,y4)[1]) and (mouse_y < coeffDr(x1,y1,x4,y4)[0]*mouse_x + coeffDr(x1,y1,x4,y4)[1])
            cond2 = (mouse_y < coeffDr(X1,Y1,X2,Y2)[0]*mouse_x + coeffDr(X1,Y1,X2,Y2)[1]) and (mouse_y < coeffDr(X2,Y2,X3,Y3)[0]*mouse_x + coeffDr(X2,Y2,X3,Y3)[1]) and (mouse_y > coeffDr(X3,Y3,X4,Y4)[0]*mouse_x + coeffDr(X3,Y3,X4,Y4)[1]) and (mouse_y > coeffDr(X1,Y1,X4,Y4)[0]*mouse_x + coeffDr(X1,Y1,X4,Y4)[1])
        if self.type == 'quarte':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2 + self.L), (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2 + self.L)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2 - self.L), (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2 - self.L)
            cond1 = (self.x - self.R/2 < mouse_x < self.x + self.R/2) and ((self.y + self.R*np.sqrt(3)/2 < mouse_y < self.y + self.R*np.sqrt(3)/2 + self.L) or (self.y - self.R*np.sqrt(3)/2 - self.L < mouse_y < self.y - self.R*np.sqrt(3)/2))
            cond2 = False

        if not mouse_found:
            if cond1 or cond2:
                print('\nAccord {} [{} - {} - {}]'.format(self.type, self.pitches[0], self.pitches[1], self.pitches[2]) + '\nHarmonicChange: {}\nDiffConcordance: {}\nDiffRoughness: {}'.format(int(self.harmonicChange), int(self.diffConcordance), int(self.diffRoughness)))
                mouse_found = True



    def click(self):
        global stream1, image_score, tot_harmonicChange, tot_diffConcordance, tot_diffRoughness
        def coeffDr(x1,y1,x2,y2):
            α = (y1-y2)/(x1-x2)
            β = (x1*y2 - x2*y1)/(x1-x2)
            return α, β

        if self.type == 'dim':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x + self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x + self.R, self.y), (self.x + self.R + self.L*np.sqrt(3)/2,self.y + self.L/2), (self.x + self.R/2 + self.L*np.sqrt(3)/2,self.y + self.R*np.sqrt(3)/2 + self.L/2)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x - self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x - self.R, self.y), (self.x - self.R - self.L*np.sqrt(3)/2,self.y - self.L/2), (self.x - self.R/2 - self.L*np.sqrt(3)/2,self.y - self.R*np.sqrt(3)/2 - self.L/2)
            cond1 = (mouse_y > coeffDr(x1,y1,x2,y2)[0]*mouse_x + coeffDr(x1,y1,x2,y2)[1]) and (mouse_y > coeffDr(x2,y2,x3,y3)[0]*mouse_x + coeffDr(x2,y2,x3,y3)[1]) and (mouse_y < coeffDr(x3,y3,x4,y4)[0]*mouse_x + coeffDr(x3,y3,x4,y4)[1]) and (mouse_y < coeffDr(x1,y1,x4,y4)[0]*mouse_x + coeffDr(x1,y1,x4,y4)[1])
            cond2 = (mouse_y < coeffDr(X1,Y1,X2,Y2)[0]*mouse_x + coeffDr(X1,Y1,X2,Y2)[1]) and (mouse_y < coeffDr(X2,Y2,X3,Y3)[0]*mouse_x + coeffDr(X2,Y2,X3,Y3)[1]) and (mouse_y > coeffDr(X3,Y3,X4,Y4)[0]*mouse_x + coeffDr(X3,Y3,X4,Y4)[1]) and (mouse_y > coeffDr(X1,Y1,X4,Y4)[0]*mouse_x + coeffDr(X1,Y1,X4,Y4)[1])

        if self.type == 'aug':
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = (self.x - self.R/2,self.y + self.R*np.sqrt(3)/2), (self.x - self.R, self.y), (self.x - self.R - self.L*np.sqrt(3)/2,self.y + self.L/2), (self.x - self.R/2 - self.L*np.sqrt(3)/2,self.y + self.R*np.sqrt(3)/2 + self.L/2)
            (X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4) = (self.x + self.R/2,self.y - self.R*np.sqrt(3)/2), (self.x + self.R, self.y), (self.x + self.R + self.L*np.sqrt(3)/2,self.y - self.L/2), (self.x + self.R/2 + self.L*np.sqrt(3)/2,self.y - self.R*np.sqrt(3)/2 - self.L/2)
            cond1 = (mouse_y > coeffDr(x1,y1,x2,y2)[0]*mouse_x + coeffDr(x1,y1,x2,y2)[1]) and (mouse_y > coeffDr(x2,y2,x3,y3)[0]*mouse_x + coeffDr(x2,y2,x3,y3)[1]) and (mouse_y < coeffDr(x3,y3,x4,y4)[0]*mouse_x + coeffDr(x3,y3,x4,y4)[1]) and (mouse_y < coeffDr(x1,y1,x4,y4)[0]*mouse_x + coeffDr(x1,y1,x4,y4)[1])
            cond2 = (mouse_y < coeffDr(X1,Y1,X2,Y2)[0]*mouse_x + coeffDr(X1,Y1,X2,Y2)[1]) and (mouse_y < coeffDr(X2,Y2,X3,Y3)[0]*mouse_x + coeffDr(X2,Y2,X3,Y3)[1]) and (mouse_y > coeffDr(X3,Y3,X4,Y4)[0]*mouse_x + coeffDr(X3,Y3,X4,Y4)[1]) and (mouse_y > coeffDr(X1,Y1,X4,Y4)[0]*mouse_x + coeffDr(X1,Y1,X4,Y4)[1])

        if self.type == 'quarte':
            cond1 = (self.x - self.R/2 < mouse_x < self.x + self.R/2) and ((self.y + self.R*np.sqrt(3)/2 < mouse_y < self.y + self.R*np.sqrt(3)/2 + self.L) or (self.y - self.R*np.sqrt(3)/2 - self.L < mouse_y < self.y - self.R*np.sqrt(3)/2))
            cond2 = False

        if cond1 or cond2:
            print('\n................')
            print('Nouvel accord {} : [{}-{}-{}]'.format(self.type, self.pitches[0], self.pitches[1], self.pitches[2]))
            tot_harmonicChange.append(self.harmonicChange)
            tot_diffConcordance.append(self.diffConcordance)
            tot_diffRoughness.append(self.diffRoughness)

            # Ajout de l'accord à stream1
            def mod(s):
                if s[-1] == 'b': s = s[:-1] + '-'
                s += '4'
                return s
            c = chord.Chord([mod(pitch) for pitch in self.pitches])
            stream1.append(c)

            # Partition
            global image_score
            title = 'accord'
            filepath = '/Users/manuel/Dropbox (TMG)/Thèse/Tonnetz/Partition/'
            out_filepath = conv_musicxml.write(stream1, 'musicxml', fp=filepath + title + '.XML', subformats=['png'])
            os.remove(filepath + title + '.XML')
            image_score = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Tonnetz/Partition/accord-1.png')

            self.selec = True
            return True
        else: return False







class Grid:
    def __init__(self,first_pitch,x, y, nb_col=5, hauteur=6, d = 100, R = 20, L = 30):
        self.coord = []
        self.pitches = []
        self.chords = []
        self.chords_dim = []
        self.chords_aug = []
        self.chords_quarte = []
        self.nb_col = nb_col
        self.hauteur = hauteur
        self.x = x
        self.y = y
        self.d = d
        self.R = R
        self.L = L




        i0 = int(nb_col/2)
        ind0 = pitch_liste.index(first_pitch)
        for i in range(nb_col):
            l = []
            haut_col = hauteur - abs(i-i0)
            if i-i0 >= 0:
                first_pitch_index = (ind0 + (i-i0)*geom[1]) % 12
            else:
                first_pitch_index = (ind0 - (i-i0)*geom[0]) % 12
            for j in range(haut_col):
                l.append(pitch_liste[(first_pitch_index + j*geom[2]) % 12])
            self.pitches.append(l)

    def Fill_coord(self):
        i0 = int(self.nb_col/2)
        for i in range(self.nb_col):
            l_coord = []
            coord0 = (self.x + (i-i0)*self.d*np.sqrt(3)/2, self.y - abs((i-i0))*self.d/2)
            haut_col = self.hauteur - abs(i-i0)
            for j in range(haut_col):
                l_coord.append((coord0[0], coord0[1] - j*self.d))
            self.coord.append(l_coord)

    def Fill_chords(self):
        for i,p in enumerate(self.pitches[0][:-1]):
            self.chords.append(Chord(p,'major',self.coord[0][i][0], self.coord[0][i][1],self.d, self.R))
        for i,p in enumerate(self.pitches[-1][:-1]):
            self.chords.append(Chord(p,'minor',self.coord[-1][i][0], self.coord[-1][i][1],self.d, self.R))
        for col in range(1,self.nb_col-1):
            for i,p in enumerate(self.pitches[col][:-1]):
                self.chords.append(Chord(p,'minor',self.coord[col][i][0], self.coord[col][i][1],self.d, self.R))
                self.chords.append(Chord(p,'major',self.coord[col][i][0], self.coord[col][i][1],self.d, self.R))

    def Fill_chords2(self):
        self.chords_dim, self.chords_aug, self.chords_quarte = [],[],[]
        # Accords diminués
        if plot_dim:
            for col in range(1,int(self.nb_col/2)):
                for i,p in enumerate(self.pitches[col][:-1]):
                    self.chords_dim.append(Chord2(p,'dim',self.coord[col][i][0], self.coord[col][i][1], self.R, self.L))
            for col in range(int(self.nb_col/2)+1,self.nb_col-1):
                for i,p in enumerate(self.pitches[col][1:]):
                    self.chords_dim.append(Chord2(p,'dim',self.coord[col][i+1][0], self.coord[col][i+1][1], self.R, self.L))
            for i,p in enumerate(self.pitches[int(self.nb_col/2)][1:-1]):
                self.chords_dim.append(Chord2(p,'dim',self.coord[int(self.nb_col/2)][i+1][0], self.coord[int(self.nb_col/2)][i+1][1], self.R, self.L))

        # Accords augmentés
        if plot_aug:
            for col in range(1,int(self.nb_col/2)):
                for i,p in enumerate(self.pitches[col][1:]):
                    self.chords_dim.append(Chord2(p,'aug',self.coord[col][i+1][0], self.coord[col][i+1][1], self.R, self.d - self.R*np.sqrt(3)))
            for col in range(int(self.nb_col/2)+1,self.nb_col-1):
                for i,p in enumerate(self.pitches[col][:-1]):
                    self.chords_dim.append(Chord2(p,'aug',self.coord[col][i][0], self.coord[col][i][1], self.R, self.d - self.R*np.sqrt(3)))
            for i,p in enumerate(self.pitches[int(self.nb_col/2)][1:-1]):
                self.chords_dim.append(Chord2(p,'aug',self.coord[int(self.nb_col/2)][i+1][0], self.coord[int(self.nb_col/2)][i+1][1], self.R, self.d - self.R*np.sqrt(3)))

        # Accords de quarte
        if plot_quarte:
            for col in range(self.nb_col):
                for i,p in enumerate(self.pitches[col][1:-1]):
                    self.chords_dim.append(Chord2(p,'quarte',self.coord[col][i+1][0], self.coord[col][i+1][1], self.R, self.L))

    def CalculDescriptors(self):
        l_harmonicChange = []
        l_diffConcordance = []
        l_diffRoughness = []
        dic_descr = {}
        nb_chords = len(self.chords) + len(self.chords_dim) + len(self.chords_aug) + len(self.chords_quarte)
        compte = 1
        def mod(s):
            if s[-1] == 'b': s = s[:-1] + '-'
            s += '4'
            return s

        # Descripteurs sur self.chords
        for ch in self.chords:
            tuple_ind = tuple(sorted([ch.pitches[0].pitch, ch.pitches[1].pitch, ch.pitches[2].pitch]))
            if tuple_ind in dic_descr:
                l_harmonicChange.append(l_harmonicChange[dic_descr[tuple_ind]])
                l_diffConcordance.append(l_diffConcordance[dic_descr[tuple_ind]])
                l_diffRoughness.append(l_diffRoughness[dic_descr[tuple_ind]])
            else:
                dic_descr[tuple_ind] = len(l_harmonicChange)
                ch.stream = copy.deepcopy(stream1)
                ch.stream.append(chord.Chord([mod(ch.pitches[0].pitch), mod(ch.pitches[1].pitch), mod(ch.pitches[2].pitch)]))
                l = ListeAccords(ch.stream, instr = (K,decr,σ), memory_size = L_mem, memory_decr_ponderation = decr_mem)
                l.HarmonicDescriptors()
                l_harmonicChange.append(l.Liste('harmonicChange')[-1])
                l_diffConcordance.append(l.Liste('diffConcordanceContext')[-1])
                l_diffRoughness.append(l.Liste('diffRoughnessContext')[-1])
            print('Accord {}/{} : fini'.format(compte, nb_chords))
            compte += 1

        for ch2 in self.chords_dim + self.chords_aug + self.chords_quarte:
            if tuple(sorted(ch2.pitches)) in dic_descr:
                l_harmonicChange.append(l_harmonicChange[dic_descr[tuple(sorted(ch2.pitches))]])
                l_diffConcordance.append(l_diffConcordance[dic_descr[tuple(sorted(ch2.pitches))]])
                l_diffRoughness.append(l_diffRoughness[dic_descr[tuple(sorted(ch2.pitches))]])
            else:
                dic_descr[tuple(sorted(ch2.pitches))] = len(l_harmonicChange)
                ch2.stream = copy.deepcopy(stream1)
                ch2.stream.append(chord.Chord([mod(ch2.pitches[0]), mod(ch2.pitches[1]), mod(ch2.pitches[2])]))
                l = ListeAccords(ch2.stream, instr = (K,decr,σ), memory_size = L_mem, memory_decr_ponderation = decr_mem)
                l.HarmonicDescriptors()
                l_harmonicChange.append(l.Liste('harmonicChange')[-1])
                l_diffConcordance.append(l.Liste('diffConcordanceContext')[-1])
                l_diffRoughness.append(l.Liste('diffRoughnessContext')[-1])
            print('Accord {}/{} : fini'.format(compte, nb_chords))
            compte += 1


        max_harmCh, max_diffConc, max_diffRough = max(l_harmonicChange), max(l_diffConcordance), max(l_diffRoughness)
        min_harmCh, min_diffConc, min_diffRough = min(l_harmonicChange), min(l_diffConcordance), min(l_diffRoughness)
        δ = 0
        if max_harmCh-min_harmCh != 0: l_harmonicChange = [((100-δ)/(max_harmCh-min_harmCh))*(elt-max_harmCh) + 100 for elt in l_harmonicChange]
        if max_diffConc-min_diffConc != 0: l_diffConcordance = [((100-δ)/(max_diffConc-min_diffConc))*(elt-max_diffConc) + 100 for elt in l_diffConcordance]
        if max_diffRough-min_diffRough != 0: l_diffRoughness = [((100-δ)/(max_diffRough-min_diffRough))*(elt-max_diffRough) + 100 for elt in l_diffRoughness]
        for i,ch in enumerate(self.chords + self.chords_dim + self.chords_aug + self.chords_quarte):
            ch.harmonicChange = l_harmonicChange[i]
            ch.diffConcordance = l_diffConcordance[i]
            ch.diffRoughness = l_diffRoughness[i]
        print('Calcul fini')


    def show(self):
        global mouse_found
        mouse_found = False

        for chord in self.chords_dim + self.chords_aug + self.chords_quarte:
            chord.show()
        for chord in self.chords:
            chord.show()



    def draw(self):
        for chord in self.chords:
            chord.draw()
        for chord in self.chords_dim + self.chords_aug + self.chords_quarte:
            chord.draw()



        #Bouton save
        stroke(0)
        fill(95,95,100,250)
        rect((largeur-90, 0), 90, 25)
        # stroke(255,255)
        text_font(f0)
        fill(0)
        text_align("RIGHT")
        p5.text('Enregistrer',(largeur-10,5))

        #Bouton paramètres
        fill(95,95,100,250)
        rect((0, 0), 160, 25)
        # stroke(255,255)
        text_font(f0)
        fill(0)
        text_align("LEFT")
        p5.text('Paramètres d\'affichage',(5,5))


    def click(self):
        found = False
        for ch in self.chords_dim + self.chords_aug + self.chords_quarte:
            if not found:
                change = ch.click()
                if change:
                    print('\nNouvel accord {} : [{} - {} - {}]'.format(ch.type, ch.pitches[0], ch.pitches[1], ch.pitches[2]))
                    print('Calcul des descripteurs...')
                    self.CalculDescriptors()
                    found = True

        for ch in self.chords:
            if not found:
                change = ch.click()
                if change:
                    print('\nNouvel accord {}: [{} - {} - {}]'.format(ch.type, ch.pitches[0].pitch, ch.pitches[1].pitch, ch.pitches[2].pitch))
                    print('Calcul des descripteurs...')
                    self.CalculDescriptors()
                    found = True





class InterfaceCouleurs(Frame):

    def __init__(self, window, space, row_start, column_start, spaceNames):
        Frame.__init__(self, window, relief=RIDGE, borderwidth=3)
        self.grid(column = column_start, row = row_start, columnspan = 6, rowspan = len(space) + 1, padx = 50, pady = 10)

        self.colLab = Label(self, text = 'Couleurs')
        self.colLab.configure(font= 'Arial 15')
        self.colLab.grid(column = column_start, row = row_start)

        self.rouge = Label(self, text = 'Rouge', foreground = 'red')
        self.vert = Label(self, text = 'Vert', foreground = 'green')
        self.bleu = Label(self, text = 'Bleu', foreground = 'blue')
        self.jet = Label(self, text = 'Arc-en-ciel', foreground = 'black')
        self.rouge.grid(column = column_start + 2, row = row_start)
        self.vert.grid(column = column_start + 3, row = row_start)
        self.bleu.grid(column = column_start + 4, row = row_start)
        self.jet.grid(column = column_start + 5, row = row_start)

        self.varR = StringVar(None, colorR)
        self.varG = StringVar(None, colorG)
        self.varB = StringVar(None, colorB)
        self.varJet = StringVar(None, colorJet)



        def clickedR():
            print('Rouge : ' + self.varR.get())
            self.varJet.set('defaut')
        def clickedG():
            print('Vert : ' + self.varG.get())
            self.varJet.set('defaut')
        def clickedB():
            print('Bleu : ' + self.varG.get())
            self.varJet.set('defaut')
        def clickedJet():
            print('Arc-en-ciel : ' + self.varJet.get())
            self.varR.set('defaut')
            self.varG.set('defaut')
            self.varB.set('defaut')


        row_count = 1
        for i,descr in enumerate(space):
            self.lab_descr = Label(self, text = spaceNames[i])
            self.lab_descr.grid(column = column_start+1, row = row_start + row_count)

            self.radR = Radiobutton(self, value=descr, variable=self.varR, command=clickedR)
            self.radG = Radiobutton(self, value=descr, variable=self.varG, command=clickedG)
            self.radB = Radiobutton(self, value=descr, variable=self.varB, command=clickedB)
            self.radJet = Radiobutton(self, value=descr, variable=self.varJet, command=clickedJet)

            self.radR.grid(column=column_start + 2, row=row_start + row_count)
            self.radG.grid(column=column_start + 3, row=row_start + row_count)
            self.radB.grid(column=column_start + 4, row=row_start + row_count)
            self.radJet.grid(column=column_start + 5, row=row_start + row_count)

            row_count += 1



def Parametres():
    # Création de l'objet fenêtre tk
    window = Tk()
    window.title("Paramètres d'affichage")
    window.geometry('530x500')

    # Types d'accords
    frame_acc = Frame(window, relief=RIDGE, borderwidth=3)
    frame_acc.grid(column = 0+1, row = 0, columnspan = 3, rowspan = 4, padx = 70, pady = 10)

    accLab = Label(frame_acc, text = 'Type d\'accords')
    accLab.configure(font= 'Arial 15')

    Lab_dim = Label(frame_acc, text = 'Diminués')
    Lab_aug = Label(frame_acc, text = 'Augmentés')
    Lab_quarte = Label(frame_acc, text = 'Quartes superposées')
    Var_dim = BooleanVar(None, plot_dim)
    Var_aug = BooleanVar(None, plot_aug)
    Var_quarte = BooleanVar(None, plot_quarte)
    But_dim = Checkbutton(frame_acc, variable=Var_dim)
    But_aug = Checkbutton(frame_acc, variable=Var_aug)
    But_quarte = Checkbutton(frame_acc, variable=Var_quarte)

    accLab.grid(column = 0, row = 0)
    Lab_dim.grid(row = 1, column = 1)
    Lab_aug.grid(row = 2, column = 1)
    Lab_quarte.grid(row = 3, column = 1)
    But_dim.grid(row = 1, column = 2)
    But_aug.grid(row = 2, column = 2)
    But_quarte.grid(row = 3, column = 2)

    # Couleurs
    interface = InterfaceCouleurs(window, ['harmonicChange', 'diffRoughness', 'diffConcordance'], 4, 0, ['Changement harmonique', 'Rugosité différentielle','Concordance différentielle'])

    # Timbre
    frame_tim = Frame(window, relief=RIDGE, borderwidth=3)
    frame_tim.grid(column = 0+1, row = 10, columnspan = 3, rowspan = 4, padx = 70, pady = 10)

    timLab = Label(frame_tim, text = 'Spectre')
    timLab.configure(font= 'Arial 15')

    Lab_K = Label(frame_tim, text = 'Nombre de partiels')
    Lab_decr = Label(frame_tim, text = 'décroissance')
    Lab_σ = Label(frame_tim, text = 'facteur σ')
    txt_K = Entry(frame_tim, width=10)
    txt_decr = Entry(frame_tim, width=10)
    txt_σ = Entry(frame_tim, width=10)
    txt_K.insert(0, K)
    txt_decr.insert(0, decr)
    txt_σ.insert(0, σ)

    timLab.grid(row = 10, column = 0)
    Lab_K.grid(row = 11, column = 1)
    Lab_decr.grid(row = 12, column = 1)
    Lab_σ.grid(row = 13, column = 1)
    txt_K.grid(row = 11, column = 2)
    txt_decr.grid(row = 12, column = 2)
    txt_σ.grid(row = 13, column = 2)

    # Mémoire
    frame_mem = Frame(window, relief=RIDGE, borderwidth=3)
    frame_mem.grid(column = 0+1, row = 14, columnspan = 3, rowspan = 3, padx = 100, pady = 10)
    memLab = Label(frame_mem, text = 'Mémoire')
    memLab.configure(font= 'Arial 15')

    Lab_L= Label(frame_mem, text = 'Longueur')
    Lab_decrMem = Label(frame_mem, text = 'décroissance')
    txt_L = Entry(frame_mem, width=10)
    txt_decrMem = Entry(frame_mem, width=10)
    txt_L.insert(0, L_mem)
    txt_decrMem.insert(0, decr_mem)

    memLab.grid(row = 14, column = 0)
    Lab_L.grid(row = 15, column = 1)
    Lab_decrMem.grid(row = 16, column = 1)
    txt_L.grid(row = 15, column = 2)
    txt_decrMem.grid(row = 16, column = 2)

    # Appliquer
    def clickOk():
        NouvCalc = False
        global colorR, colorG, colorB, colorJet, K, decr, σ, L_mem, decr_mem, plot_dim, plot_aug, plot_quarte
        global grid

        if (int(txt_K.get())!=int(K)) or (float(txt_decr.get())!=float(decr)) or (float(txt_σ.get())!=float(σ)) or (int(txt_L.get())!=int(L_mem)) or (float(txt_decrMem.get())!=float(decr_mem)) or (Var_dim.get()!=plot_dim) or (Var_aug.get()!=plot_aug) or (Var_quarte.get()!=plot_quarte):
            NouvCalc = True
        if (Var_dim.get()!=plot_dim) or (Var_aug.get()!=plot_aug) or (Var_quarte.get()!=plot_quarte):
            plot_dim, plot_aug, plot_quarte = Var_dim.get(), Var_aug.get(), Var_quarte.get()
            grid.Fill_chords2()
            # print(g)
        colorR, colorG, colorB, colorJet = interface.varR.get(),  interface.varG.get(), interface.varB.get(), interface.varJet.get()
        K, decr, σ = int(txt_K.get()),  float(txt_decr.get()), float(txt_σ.get())
        L_mem, decr_mem = int(txt_L.get()),  float(txt_decrMem.get())
        window.withdraw()
        window.quit()
        if NouvCalc and len(stream1)!=0:
            grid.CalculDescriptors()



    ok = Button(window,text = 'Appliquer',command = clickOk)
    ok.grid(column = 1, row = 17, padx=150, pady=10)


    return window



if __name__ == '__main__':
    window = Parametres()
    run()
