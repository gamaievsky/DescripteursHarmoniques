import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from p5 import *

sys.path.append(".")
from classes import ListeAccords, Accord


with open('liste_interval_vectors.pkl', 'rb') as f:
    liste_interval_vectors = pickle.load(f)
dic_interv_vect = {i:liste_interval_vectors[i-1] for i in range(1,77+1)}


######## LOAD CONCORDANCE AND ROUGHNESS

# from music21 import *
# from operator import itemgetter, attrgetter
# import tunings
# import parametres
# listeSpectresAccords = []


# score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/all_iv_chords.musicxml')
#
# l = ListeAccords(score)
# l.HarmonicDescriptors()
# liste_concordance = l.Liste('concordance')
# liste_roughness = l.Liste('roughness')
# max_conc, max_rough = max(liste_concordance), max(liste_roughness)
# liste_concordance = [round(100*float(l)/max_conc,2) for l in liste_concordance]
# liste_roughness = [round(100*float(l)/max_rough,2) for l in liste_roughness]
#
# dic_concordance = {i:liste_concordance[i-1] for i in range(1,77+1)}
# dic_roughness = {i:liste_roughness[i-1] for i in range(1,77+1)}
#
# with open('dic_concordance_normNo.pkl', 'wb') as f:
#     pickle.dump(dic_concordance, f)
# with open('dic_roughness_normNo.pkl', 'wb') as f:
#     pickle.dump(dic_roughness, f)




######### DRAW

with open('dic_concordance_normTot.pkl', 'rb') as f:
    dic_concordance = pickle.load(f)

with open('dic_roughness_normTot.pkl', 'rb') as f:
    dic_roughness = pickle.load(f)

with open('dic_img.pkl', 'rb') as f:
    img_dic = pickle.load(f)

dic_ind_start = {i:[1,2,8,20,35,48,59,66,71,74,76,77][i-1] for i in range(1,12+1)}
dic_ind_end = {i:[1,7,19,34,47,58,65,70,73,75,76,77][i-1] for i in range(1,12+1)}

# dic_concordance = {ind : np.log(1 + dic_concordance[ind]) * 100./np.log(101) for ind in dic_concordance}
# dic_roughness = {ind : (np.exp(dic_roughness[ind]) - 1) * 100./(np.exp(100)-1) for ind in dic_roughness}

# print('Concordance')
# print([int(dic_concordance[ind]) for ind in dic_concordance])
# print(np.std([dic_concordance[ind] for ind in dic_concordance]))
# print('Roughness')
# print([int(dic_roughness[ind]) for ind in dic_roughness])
# print(np.std([dic_roughness[ind] for ind in dic_roughness]))
# print('\nLoi uniforme : ' + str(np.std([i*100/77 for i in range(1,78)])))


def setup():
    size(1200, 800)
    color_mode('RGBA', 100)
    global f
    f = create_font("Arial.ttf", 16) # STEP 2 Create Font
    global grid
    grid = Grid()


def draw():
    background(255)
    grid.draw()

def mouse_pressed():
    global found
    found = False
    grid.click()
    # save(filename='permutahedre3.png')





class Chord:
    def __init__(self, ind):
        self.ind = ind
        self.image = img_dic[self.ind]
        self.interval_vector = dic_interv_vect[ind]
        self.concordance = dic_concordance[self.ind]
        self.roughness = dic_roughness[self.ind]

    def draw(self,x,y,h, a_conc, b_conc, a_rgh, b_rgh):
        α = self.image.width / self.image.height

        stroke(0)
        if self.ind == 1: no_fill()
        else:
            fill(a_rgh * self.roughness + b_rgh, 0, a_conc * self.concordance + b_conc,180)
        rect((x, y), α*h, h)

        image(self.image, (x, y), (α*h,h))



    def click(self,x,y,w,h):
        if (x < mouse_x < x+w) and (y < mouse_y < y+h):
            print('Accord {}: {}'.format(self.ind, self.interval_vector) + '\n' + 'Concordance: {}, roughness: {}'.format(dic_concordance[self.ind], dic_roughness[self.ind]))


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
        conc_min = min([dic_concordance[ind] for ind in indices])
        conc_max = max([dic_concordance[ind] for ind in indices])
        rgh_min = min([dic_roughness[ind] for ind in indices])
        rgh_max = max([dic_roughness[ind] for ind in indices])
        if self.liste_column[0].ind == 1:  δ = 0
        else: δ = 10
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


    def click(self, h_ch = 45, s_w = 10, s = 3):
        i = 0
        while (not found) and (i < self.n):
            self.liste_column[i].click(liste_x_crt[i], liste_α[i] * h_ch , h_ch, s)
            i += 1







if __name__ == '__main__':
    run()
