import numpy as np
# import matplotlib.pyplot as plt
import pickle
import sys
import os
from p5 import *
from tkinter import *
import tkinter as tk
from tkinter.ttk import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import random
import sys
from threading import Thread
import time



sys.path.append(".")


######### DRAW

K, decr, σ = 11, 1/2, 0.005

# Dictionnaire des descripteurs harmoniques
with open('Dic_Harm.pkl', 'rb') as f:
    Dic_Harm = pickle.load(f)
# Dictionnaire des représentations réduites
with open('Dic_iv.pkl', 'rb') as f:
    Dic_iv = pickle.load(f)
# Dictionnaire des cardinalités des classes premières
with open('Dic_card.pkl', 'rb') as f:
    Dic_card = pickle.load(f)
# Dictionnaire des images
with open('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/Dic_img.pkl', 'rb') as f:
    Dic_img = pickle.load(f)

with open('numSave.pkl', 'rb') as f:
    numSave = pickle.load(f)


dic_ind_start = {i:[1,2,8,20,35,48,59,66,71,74,76,77][i-1] for i in range(1,12+1)}
dic_ind_end = {i:[1,7,19,34,47,58,65,70,73,75,76,77][i-1] for i in range(1,12+1)}


colorR1, colorG1, colorB1, colorJet1, colorR2, colorG2, colorB2, colorJet2 = 'defaut', 'defaut', 'defaut','roughness', 'concordanceOrdre3', 'concordanceTotale', 'harmonicity','defaut'
rel = True
niveau2 = False
spectre_change = False

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)
COL = MplColorHelper('jet', 10, 100)



def setup():
    size(1200, 800)
    title('Classification des accords - timbre (K: {}, decr: {}, σ: {})'.format(K,decr,σ))
    color_mode('RGBA', 100)
    global f
    f = create_font("Arial.ttf", 14)

    global grid
    grid = Grid()
    global grid2


def draw():
    global niveau2
    global grid, grid2
    global K,decr,σ,spectre_change
    background(255)
    grid.draw()

    if niveau2:
            global x,y,w,h,R,h_ch
            grid2 = Grid2(niveau2_ind, concordance, roughness)
            # Calcul des paramètres d'affichage
            x, y, w, h = 400,50,700,700
            R0, h_ch0 = 200, 100
            if grid2.card < 6:
                R, h_ch = R0, h_ch0
            else:
                α = grid2.liste_chords2[0].image.width / grid2.liste_chords2[0].image.height
                δ = np.tan(2*np.pi / grid2.card) / (α + np.tan(2*np.pi / grid2.card))
                if δ*R0 > 100: R, h_ch = R0, h_ch0
                else:
                    R = w / (2 + α*δ) - 10
                    h_ch = R*δ

            stroke(0,200)
            stroke_weight(2)
            line((niv2_x,niv2_y), (x,y))
            line((niv2_x,niv2_y+niv2_h), (x,y+h))
            line((niv2_x+niv2_w,niv2_y), (x+w,y))
            line((niv2_x+niv2_w,niv2_y+niv2_h), (x+w,y+h))

            grid2.draw(x,y,w,h,R,h_ch)

    if spectre_change:
        for column in grid.liste_column:
            for chord in column.liste_chords:
                chord.concordance = Dic_Harm[(K,decr,σ)]['concordance'][chord.ind]
                chord.roughness = Dic_Harm[(K,decr,σ)]['roughness'][chord.ind]
        title('Classification des accords - timbre (K: {}, decr: {}, σ: {})'.format(K,decr,σ))
        spectre_change = False




def mouse_pressed():
    global numSave

    # Bouton Paramètres
    if (0 < mouse_x < 160) and (0 < mouse_y < 25):
        no_loop()
        window.deiconify()
        window.mainloop()
        loop()
    # Bouton Save
    if (1110 < mouse_x < 1200) and (0 < mouse_y < 25):
        stroke(250)
        fill(250)
        rect((1110, 0), 90, 25)
        fill(250)
        rect((0, 0), 160, 25)
        save(filename='/Users/manuel/Dropbox (TMG)/Thèse/Estrada/Captures/permutahedre{}.png'.format(int(numSave)))
        print('file {} saved'.format(numSave))
        numSave += 1
        with open('numSave.pkl', 'wb') as f:
            pickle.dump(numSave, f)


    global niveau2
    if not niveau2:
        global found
        found = False
        grid.click()
    else:
        global found2
        found2 = False
        grid2.click(x,y,w,h,R,h_ch)




class Chord:
    def __init__(self, ind):
        self.ind = ind
        self.image = Dic_img[self.ind]
        self.interval_vector = Dic_iv[ind]
        self.concordance = Dic_Harm[(K,decr,σ)]['concordance'][self.ind]
        self.roughness = Dic_Harm[(K,decr,σ)]['roughness'][self.ind]
        self.dic_color = {}



    def draw(self,x,y,h, a_conc, b_conc, a_rgh, b_rgh):
        α = self.image.width / self.image.height
        var = [colorR1,colorG1,colorB1,colorJet1]
        for i,col in enumerate(['R','G','B','Jet']):
            if var[i] == 'roughness':
                self.dic_color[col]= a_rgh**rel * self.roughness + b_rgh*rel
            elif var[i] == 'concordance':
                self.dic_color[col] = a_conc**rel * self.concordance + b_conc*rel
            else: self.dic_color[col] = 0

        stroke(0)
        if niveau2 and (niveau2_ind == self.ind):
            stroke_weight(3)
        else: stroke_weight(1)

        if self.ind == 1: no_fill()
        else:
            # no_fill()
            if [colorR1, colorG1, colorB1, colorJet1] == ['defaut','defaut','defaut','defaut']:
                fill(255,150)
            elif colorJet1 == 'defaut':
                fill(self.dic_color['R'], self.dic_color['G'], self.dic_color['B'],150)
            else:
                z = self.dic_color['Jet']
                fill(100*COL.get_rgb(z)[0],100*COL.get_rgb(z)[1],100*COL.get_rgb(z)[2],150)
        rect((x, y), α*h, h)

        image(self.image, (x, y), (α*h,h))



    def click(self,x,y,w,h):
        global niveau2, niveau2_ind, concordance, roughness, niv2_x, niv2_y, niv2_h, niv2_w
        if not niveau2:
            if (x < mouse_x < x+w) and (y < mouse_y < y+h):
                print('Accord {}: {}'.format(self.ind, self.interval_vector) + '\n' + 'Cardinality: {}'.format(Dic_card[self.ind]) + '\n' + 'Concordance: {}\nRoughness: {}'.format(self.concordance, self.roughness) + '\n')
                niveau2 = True
                niveau2_ind = self.ind
                concordance = self.concordance
                roughness = self.roughness
                niv2_x = x
                niv2_y = y
                niv2_h = h
                niv2_w = w

class Column:
    def __init__(self, ind):
        self.ind = ind
        self.liste_chords = []
        ind_start = dic_ind_start[self.ind]
        ind_end = dic_ind_end[self.ind]
        self.n = ind_end - ind_start + 1

        for ind in range(ind_start, ind_end+1):
            self.liste_chords.append(Chord(ind))

    def draw(self, x, w_ch, h_ch, s , a_conc, b_conc, a_rgh, b_rgh):
        for i, chord in enumerate(self.liste_chords):
            y = height/2 - (self.n / 2 - i) * h_ch - ((self.n - i) / 2 - i) * s
            chord.draw(x, y, h_ch, a_conc, b_conc, a_rgh, b_rgh)

        text_font(f)
        fill(0)
        text_align("CENTER")
        text('N = {}'.format(self.ind),(x + w_ch/2.,height/2 - (self.n / 2) * h_ch - ((self.n) / 2) * s - 23))
        # text('N = {}'.format(self.ind), (x,height/2 - (self.n / 2) * h_ch - ((self.n) / 2) * s - 10))



    def click(self, x, w_ch, h_ch, s = 10):
        i = 0
        while (not found) and (i < self.n):
            y = height/2 - (self.n / 2 - i) * h_ch - ((self.n - i) / 2 - i) * s
            self.liste_chords[i].click(x,y,w_ch,h_ch)
            i+=1

class Grid:
    def __init__(self, ind_start = 1, ind_end = 12):
        self.liste_column = []
        for ind in range(ind_start, ind_end + 1):
            self.liste_column.append(Column(ind))
        self.n = ind_end - ind_start + 1


    def draw(self, h_ch = 45, x_crt = 10, s_w = 10, s = 3):
        global liste_x_crt
        liste_x_crt = [x_crt]
        global liste_α
        liste_α = []
        # dic_α = {i:[1.2,0.8,1.13,1.34,1.6,1.79,1.95,2.2,2.54,2.58,2.68,2.92][i-1] for i in range(1,12+1)}
        # dic_y = {[260,290,360,400,450,500,540,580,630,670,720,760]
        # Coefficients pour le dessin
        indices = range(dic_ind_start[self.liste_column[0].ind], dic_ind_end[self.liste_column[-1].ind] +1)
        conc_min = min([Dic_Harm[(K,decr,σ)]['concordance'][ind] for ind in indices])
        conc_max = max([Dic_Harm[(K,decr,σ)]['concordance'][ind] for ind in indices])
        rgh_min = min([Dic_Harm[(K,decr,σ)]['roughness'][ind] for ind in indices])
        rgh_max = max([Dic_Harm[(K,decr,σ)]['roughness'][ind] for ind in indices])
        if self.liste_column[0].ind == 1:  δ = 0
        else: δ = 2
        a_conc = (100-δ)/(conc_max-conc_min)
        b_conc = 100. - conc_max*((100-δ)/(conc_max-conc_min))
        a_rgh = (100-δ)/(rgh_max-rgh_min)
        b_rgh = 100. - rgh_max*((100-δ)/(rgh_max-rgh_min))
        # Dessin
        for i, column in enumerate(self.liste_column):
            liste_α.append(self.liste_column[i].liste_chords[0].image.width / self.liste_column[i].liste_chords[0].image.height)
            column.draw(x_crt, h_ch * liste_α[i], h_ch, s, a_conc, b_conc, a_rgh, b_rgh)
            x_crt +=  h_ch * liste_α[i] + s_w
            liste_x_crt.append(x_crt)

        #Bouton save
        stroke(0)
        fill(95,95,100,250)
        rect((1110, 0), 90, 25)
        # stroke(255,255)
        text_font(f)
        fill(0)
        text_align("RIGHT")
        text('Enregistrer',(1190,5))

        #Bouton paramètres
        fill(95,95,100,250)
        rect((0, 0), 160, 25)
        # stroke(255,255)
        text_font(f)
        fill(0)
        text_align("LEFT")
        text('Paramètres d\'affichage',(5,5))




    def click(self, h_ch = 45, s_w = 10, s = 3):
        i = 0
        while (not found) and (i < self.n):
            self.liste_column[i].click(liste_x_crt[i], liste_α[i] * h_ch , h_ch, s)
            i += 1

class Chord2:
    def __init__(self, ind, perm, concordance, roughness):
        self.ind = ind
        self.perm = perm
        self.id = '{}-{}'.format(ind, perm)
        self.image = Dic_img[self.id]
        self.interval_vector = Dic_iv[self.id]

        self.concordance = concordance
        self.roughness = roughness
        self.conc3 = Dic_Harm[(K,decr,σ)]['concordanceOrdre3'][self.id]
        self.concTot = Dic_Harm[(K,decr,σ)]['concordanceTotale'][self.id]
        self.harm = Dic_Harm[(K,decr,σ)]['harmonicity'][self.id]
        self.tension = Dic_Harm[(K,decr,σ)]['tension'][self.id]
        self.dic_color = {}


    def draw(self,x,y,h,a_conc3, b_conc3, a_concTot, b_concTot, a_harm, b_harm, a_tension, b_tension):
        α = self.image.width / self.image.height
        var = [colorR2,colorG2,colorB2,colorJet2]
        for i,col in enumerate(['R','G','B','Jet']):
            if var[i] == 'concordanceOrdre3':
                self.dic_color[col]= a_conc3**rel * self.conc3 + b_conc3*rel
            elif var[i] == 'concordanceTotale':
                self.dic_color[col] = a_concTot**rel * self.concTot + b_concTot*rel
            elif var[i] == 'harmonicity':
                self.dic_color[col] = a_harm**rel * self.harm + b_harm*rel
            elif var[i] == 'tension':
                self.dic_color[col] = a_tension**rel * self.tension + b_tension*rel
            else: self.dic_color[col] = 0


        stroke(0)
        if [colorR2, colorG2, colorB2, colorJet2] == ['defaut','defaut','defaut','defaut']:
            fill(255,150)
        elif colorJet2 == 'defaut':
            fill(self.dic_color['R'], self.dic_color['G'], self.dic_color['B'],150)
        else:
            z = self.dic_color['Jet']
            fill(100*COL.get_rgb(z)[0],100*COL.get_rgb(z)[1],100*COL.get_rgb(z)[2],150)
        rect((x, y), α*h, h)
        image(self.image, (x, y), (α*h,h))


    def click(self,x,y,h):
        α = self.image.width/self.image.height
        if (x < mouse_x < x+α*h) and (y < mouse_y < y+h):
            print('Accord ' + self.id + ': {}'.format(self.interval_vector) + '\nConcordance3: {}\nConcordanceTot: {}\nHarmonicity: {}\nTension: {}\n'.format(round(self.conc3,2), round(self.concTot,2), round(self.harm,2), round(self.tension,2)))

class Grid2:
    def __init__(self, ind, concordance, roughness):
        self.ind = ind
        self.interval_vector = Dic_iv[ind]
        self.card = Dic_card[ind]
        self.concordance = concordance
        self.roughness = roughness
        self.liste_chords2 = []
        for perm in range(1,self.card + 1):
            self.liste_chords2.append(Chord2(ind, perm, self.concordance, self.roughness))


    def draw(self,x,y,w,h,R,h_ch):
        stroke(0)
        fill(100,230)
        rect((400, 50),700, 700)
        conc3_min, conc3_max = min([chord2.conc3  for chord2 in self.liste_chords2]), max([chord2.conc3  for chord2 in self.liste_chords2])
        concTot_min, concTot_max = min([chord2.concTot  for chord2 in self.liste_chords2]), max([chord2.concTot  for chord2 in self.liste_chords2])
        harm_min, harm_max = min([chord2.harm  for chord2 in self.liste_chords2]), max([chord2.harm  for chord2 in self.liste_chords2])
        tension_min, tension_max = min([chord2.tension  for chord2 in self.liste_chords2]), max([chord2.tension  for chord2 in self.liste_chords2])
        if self.card == 1:
            a_conc3, b_conc3, a_concTot, b_concTot, a_harm, b_harm, a_tension, b_tension = 1,1,1,1,1,1,1,1
        else:
            δ = 2
            a_conc3, b_conc3  = (100-δ)/(conc3_max-conc3_min), 100. - conc3_max*((100-δ)/(conc3_max-conc3_min))
            a_concTot, b_concTot  = (100-δ)/(concTot_max-concTot_min), 100. - concTot_max*((100-δ)/(concTot_max-concTot_min))
            a_harm, b_harm  = (100-δ)/(harm_max-harm_min), 100. - harm_max*((100-δ)/(harm_max-harm_min))
            a_tension, b_tension  = (100-δ)/(tension_max-tension_min), 100. - tension_max*((100-δ)/(tension_max-tension_min))

        α = self.liste_chords2[0].image.width / self.liste_chords2[0].image.height
        x0 = x + w/2 - α*h_ch/2
        y0 = y + h/2 - h_ch/2
        for chord2 in self.liste_chords2:
            if self.card == 6:
                ordre = [0,2,5,1,4,3]
                chord2.draw(x0 + R*np.cos(-np.pi/2 + 2*np.pi*ordre[chord2.perm-1] / self.card), y0 + R*np.sin(-np.pi/2 + 2*np.pi*ordre[chord2.perm-1] / self.card), h_ch, a_conc3, b_conc3, a_concTot, b_concTot, a_harm, b_harm, a_tension, b_tension)
            else:
                chord2.draw(x0 + R*np.cos(-np.pi/2 + 2*np.pi*(chord2.perm-1) / self.card), y0 + R*np.sin(-np.pi/2 + 2*np.pi*(chord2.perm-1) / self.card), h_ch, a_conc3, b_conc3, a_concTot, b_concTot, a_harm, b_harm, a_tension, b_tension)
        text_font(f)
        fill(0)
        text_align("CENTER")
        text('Prime form {}\n{}\nCardinality: {}'.format(self.ind, self.interval_vector, self.card),(x + w/2,y + h/2))

    def click(self,x,y,w,h,R,h_ch):
        global niveau2
        if not (((x < mouse_x < x+w) and (y < mouse_y < y+h)) or ((0 < mouse_x < 160) and (0 < mouse_y < 25)) or ((1110 < mouse_x < 1200) and (0 < mouse_y < 25))):
            niveau2 = False
        else:
            α = self.liste_chords2[0].image.width / self.liste_chords2[0].image.height
            x0 = x + w/2 - α*h_ch/2
            y0 = y + h/2 - h_ch/2
            for chord2 in self.liste_chords2:
                chord2.click(x0 + R*np.cos(-np.pi/2 + 2*np.pi*(chord2.perm-1) / self.card), y0 + R*np.sin(-np.pi/2 + 2*np.pi*(chord2.perm-1) / self.card), h_ch)

class Interface(Frame):

    def __init__(self, window, space,ind, row_start, column_start):
        Frame.__init__(self, window, relief=RIDGE, borderwidth=3)
        self.grid(column = column_start, row = row_start, columnspan = 7, rowspan = len(space) + 1, padx = 50, pady = 10)

        if ind == 1: self.niveau = Label(self, text='Classes premières')
        else: self.niveau = Label(self, text='Classes normales')
        self.niveau.configure(font= 'Arial 15')#"Verana 15 underline")
        self.niveau.grid(column = column_start, row = row_start)

        self.rouge = Label(self, text = 'Rouge', foreground = 'red')
        self.vert = Label(self, text = 'Vert', foreground = 'green')
        self.bleu = Label(self, text = 'Bleu', foreground = 'blue')
        self.jet = Label(self, text = 'Arc-en-ciel', foreground = 'black')
        self.rouge.grid(column = column_start + 2, row = row_start)
        self.vert.grid(column = column_start + 3, row = row_start)
        self.bleu.grid(column = column_start + 4, row = row_start)
        self.jet.grid(column = column_start + 5, row = row_start)
        if ind==1:
            self.varR = StringVar(None, colorR1)
            self.varG = StringVar(None, colorG1)
            self.varB = StringVar(None, colorB1)
            self.varJet = StringVar(None, colorJet1)
        elif ind==2:
            self.varR = StringVar(None, colorR2)
            self.varG = StringVar(None, colorG2)
            self.varB = StringVar(None, colorB2)
            self.varJet = StringVar(None, colorJet2)


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
        for descr in space + ['defaut']:
            self.lab_descr = Label(self, text = descr[0].upper() + descr[1:])
            self.lab_descr.grid(column = column_start + 1, row = row_start + row_count)

            self.radR = Radiobutton(self, value=descr, variable=self.varR, command=clickedR)
            self.radG = Radiobutton(self, value=descr, variable=self.varG, command=clickedG)
            self.radB = Radiobutton(self, value=descr, variable=self.varB, command=clickedB)
            self.radJet = Radiobutton(self, value=descr, variable=self.varJet, command=clickedJet)


            self.radR.grid(column=column_start + 2, row=row_start + row_count)
            self.radG.grid(column=column_start + 3, row=row_start + row_count)
            self.radB.grid(column=column_start + 4, row=row_start + row_count)
            self.radJet.grid(column=column_start + 5, row=row_start + row_count)
            row_count += 1

def ParametresCouleurs():
    # Création de l'objet fenêtre tk
    window = Tk()
    window.title("Paramètres de timbre et de couleur")
    window.geometry('550x450')

    # Spectre
    global K, decr, σ
    frame_sp = Frame(window, relief=RIDGE, borderwidth=3)
    frame_sp.grid(column = 0, row = 0, columnspan = 8, rowspan = 2, padx = 100, pady = 10)
    SpLab = Label(frame_sp, text = 'Spectre')
    SpLab.configure(font= 'Arial 15')
    SpLab.grid(column = 0, row = 0)

    KLab = Label(frame_sp, text = 'K')
    KChoices = [5,7,11,17]
    KVar = IntVar(None, K)
    KMenu = tk.OptionMenu(frame_sp, KVar, *KChoices)
    KLab.grid(row = 1, column = 1)
    KMenu.grid(row = 1, column = 2)

    DecrLab = Label(frame_sp, text = '   Decr')
    DecrChoices = [0,0.5,1]
    DecrVar = DoubleVar(None, decr)
    DecrMenu = tk.OptionMenu(frame_sp, DecrVar, *DecrChoices)
    DecrLab.grid(row = 1, column = 4)
    DecrMenu.grid(row = 1, column = 5)

    SigLab = Label(frame_sp, text = '   σ')
    SigChoices = [0.005,0.01]
    SigVar = DoubleVar(None, σ)
    SigMenu = tk.OptionMenu(frame_sp, SigVar, *SigChoices)
    SigLab.grid(row = 1, column = 7)
    SigMenu.grid(row = 1, column = 8)

    # Couleurs
    interface1 = Interface(window, ['concordance', 'roughness'],1, 0+3, 0)
    interface2 = Interface(window, ['concordanceOrdre3', 'concordanceTotale', 'harmonicity', 'tension'], 2, 7+3, 0)

    # Afgfichage Relatif / Absolu
    frame_rel = Frame(window, relief=RIDGE, borderwidth=3)
    frame_rel.grid(column = 2, row = 15+3, columnspan = 3, rowspan = 1, padx = 100, pady = 10)

    # def clickRel():
    relLab = Label(frame_rel, text = 'Valeurs : ')
    relLab.configure(font= 'Arial 15')
    relVar = BooleanVar(None, True)
    relVar.set(True) #set check state
    def clickRel():
        global rel
        rel = relVar.get()
        print(rel)
    relBut1 = Radiobutton(frame_rel,text='relatives', value = True, variable=relVar,command = clickRel)
    relBut2 = Radiobutton(frame_rel, text='absolues',value = False, variable=relVar, command = clickRel)
    relLab.grid(row = 15+3, column = 0)
    relBut1.grid(row = 15+3, column = 1)
    relBut2.grid(row = 15+3, column = 2)

    # Valider
    def clickOk():
        global colorR1, colorG1, colorB1, colorJet1, colorR2, colorG2, colorB2, colorJet2
        colorR1, colorG1, colorB1, colorJet1 = interface1.varR.get(),  interface1.varG.get(), interface1.varB.get(), interface1.varJet.get()
        colorR2, colorG2, colorB2, colorJet2 = interface2.varR.get(),  interface2.varG.get(), interface2.varB.get(), interface2.varJet.get()
        global K, decr, σ, spectre_change
        if (K != KVar.get()) or (decr != DecrVar.get()) or (σ != SigVar.get()): spectre_change = True
        K, decr, σ = KVar.get(), DecrVar.get(), SigVar.get()
        window.withdraw()
        window.quit()
    ok = Button(window,text = 'Appliquer',command = clickOk)
    ok.grid(column = 3,row = 17+3, padx=150, pady=10)


    return window




if __name__ == '__main__':

    window = ParametresCouleurs()
    run()
