import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tkinter import *
from tkinter.ttk import *




with open('Dic_Harm.pkl', 'rb') as f:
    Dic_Harm = pickle.load(f)

with open('dic_roughness_normTot.pkl', 'rb') as f:
    dic_roughness_normTot = pickle.load(f)

print(Dic_Harm[(7,1/2,0.01)]['roughness'])
print('\n')
print(dic_roughness_normTot)

# with open('Dic_id.pkl', 'rb') as f:
#     Dic_id = pickle.load(f)
#
# l = []
#
# for i in range(351):
#     l.append(0)
# dic_tension = {id: l[Dic_id[id]]  for id in Dic_id}
#
# Dic_Harm[(7,1/2,0.01)]['tension'] = dic_tension
#
# with open('Dic_Harm.pkl', 'wb') as f:
#     pickle.dump(Dic_Harm, f)




#s
# class Interface(Frame):
#
#     def __init__(self, window, space,ind, row_start, column_start):
#         Frame.__init__(self, window, relief=RIDGE, borderwidth=3)
#         self.grid(column = column_start, row = row_start, columnspan = 6, rowspan = len(space) + 1, padx = 50, pady = 10)
#
#         self.niveau = Label(self, text='Niveau {}'.format(ind))
#         self.niveau.configure(font= 'Arial 15')#"Verana 15 underline")
#         self.niveau.grid(column = column_start, row = row_start)
#
#         self.rouge = Label(self, text = 'Rouge', foreground = 'red')
#         self.vert = Label(self, text = 'Vert', foreground = 'green')
#         self.bleu = Label(self, text = 'Bleu', foreground = 'blue')
#         self.rouge.grid(column = column_start + 2, row = row_start)
#         self.vert.grid(column = column_start + 3, row = row_start)
#         self.bleu.grid(column = column_start + 4, row = row_start)
#         self.varR = StringVar(None, space[0])
#         self.varG = StringVar(None, (space + ['defaut'])[2])
#         self.varB = StringVar(None, space[1])
#
#
#         def clickedR():
#             print('Rouge : ' + self.varR.get())
#         def clickedG():
#             print('Vert : ' + self.varG.get())
#         def clickedB():
#             print('Bleu : ' + self.varG.get())
#
#
#         row_count = 1
#         for descr in space + ['defaut']:
#             self.lab_descr = Label(self, text = descr[0].upper() + descr[1:])
#             self.lab_descr.grid(column = column_start + 1, row = row_start + row_count)
#
#             self.radR = Radiobutton(self, value=descr, variable=self.varR, command=clickedR)
#             self.radG = Radiobutton(self, value=descr, variable=self.varG, command=clickedG)
#             self.radB = Radiobutton(self, value=descr, variable=self.varB, command=clickedB)
#
#
#             self.radR.grid(column=column_start + 2, row=row_start + row_count)
#             self.radG.grid(column=column_start + 3, row=row_start + row_count)
#             self.radB.grid(column=column_start + 4, row=row_start + row_count)
#             row_count += 1
#


# colorR1, colorG1, colorB1, colorR2, colorG2, colorB2 = 'roughness', 'defaut', 'concordance', 'concordanceOrdre3', 'concordanceTotale', 'harmonicity'
# rel = True
#
#
# #
# class Interface(Frame):
#
#     def __init__(self, window, space,ind, row_start, column_start):
#         Frame.__init__(self, window, relief=RIDGE, borderwidth=3)
#         self.grid(column = column_start, row = row_start, columnspan = 6, rowspan = len(space) + 1, padx = 50, pady = 10)
#
#         self.niveau = Label(self, text='Niveau {}'.format(ind))
#         self.niveau.configure(font= 'Arial 15')#"Verana 15 underline")
#         self.niveau.grid(column = column_start, row = row_start)
#
#         self.rouge = Label(self, text = 'Rouge', foreground = 'red')
#         self.vert = Label(self, text = 'Vert', foreground = 'green')
#         self.bleu = Label(self, text = 'Bleu', foreground = 'blue')
#         self.rouge.grid(column = column_start + 2, row = row_start)
#         self.vert.grid(column = column_start + 3, row = row_start)
#         self.bleu.grid(column = column_start + 4, row = row_start)
#         if ind==1:
#             self.varR = StringVar(None, colorR1)
#             self.varG = StringVar(None, colorG1)
#             self.varB = StringVar(None, colorB1)
#         elif ind==2:
#             self.varR = StringVar(None, colorR2)
#             self.varG = StringVar(None, colorG2)
#             self.varB = StringVar(None, colorB2)
#
#         def clickedR():
#             print('Rouge : ' + self.varR.get())
#         def clickedG():
#             print('Vert : ' + self.varG.get())
#         def clickedB():
#             print('Bleu : ' + self.varG.get())
#
#
#         row_count = 1
#         for descr in space + ['defaut']:
#             self.lab_descr = Label(self, text = descr[0].upper() + descr[1:])
#             self.lab_descr.grid(column = column_start + 1, row = row_start + row_count)
#
#             self.radR = Radiobutton(self, value=descr, variable=self.varR, command=clickedR)
#             self.radG = Radiobutton(self, value=descr, variable=self.varG, command=clickedG)
#             self.radB = Radiobutton(self, value=descr, variable=self.varB, command=clickedB)
#
#
#             self.radR.grid(column=column_start + 2, row=row_start + row_count)
#             self.radG.grid(column=column_start + 3, row=row_start + row_count)
#             self.radB.grid(column=column_start + 4, row=row_start + row_count)
#             row_count += 1
#
# def ParametresCouleurs():
#     # Création de l'objet fenêtre tk
#     window = Tk()
#     window.title("Paramètres de couleur")
#     window.geometry('400x380')
#     interface1 = Interface(window, ['concordance', 'roughness'],1, 0, 0)
#     interface2 = Interface(window, ['concordanceOrdre3', 'concordanceTotale', 'harmonicity', 'tension'], 2, 7, 0)
#
#     frame_rel = Frame(window, relief=RIDGE, borderwidth=3)
#     frame_rel.grid(column = 0, row = 15, columnspan = 3, rowspan = 1, padx = 100, pady = 10)
#
#     # def clickRel():
#     relLab = Label(frame_rel, text = 'Valeurs : ')
#     relLab.configure(font= 'Arial 15')
#     relVar = BooleanVar(None, True)
#     relVar.set(True) #set check state
#     def clickRel():
#         rel = relVar.get()
#         print(rel)
#     relBut1 = Radiobutton(frame_rel,text='relatives', value = True, variable=relVar,command = clickRel)
#     relBut2 = Radiobutton(frame_rel, text='absolues',value = False, variable=relVar, command = clickRel)
#     relLab.grid(row = 15, column = 0)
#     relBut1.grid(row = 15, column = 1)
#     relBut2.grid(row = 15, column = 2)
#
#     def clickOk():
#         global colorR1, colorG1, colorB1, colorR2, colorG2, colorB2
#         colorR1, colorG1, colorB1 = interface1.varR.get(),  interface1.varG.get(), interface1.varB.get()
#         colorR2, colorG2, colorB2 = interface2.varR.get(),  interface2.varG.get(), interface2.varB.get()
#         window.withdraw()
#         window.quit()
#     ok = Button(window,text = 'Appliquer',command = clickOk)
#     ok.grid(row = 17, padx=150, pady=10)
#     return window
#
#
#
#
# if __name__ == '__main__':
#
#     window = ParametresCouleurs()
#     window.mainloop()





    # lbl = Label(window, text="Niveau 1",foreground = 'black', background='white')
    # lbl.grid(column=0, row=0)
    #
    # # frame2 = tk.Frame( master=window,relief=tk.RAISED,borderwidth=1)
    #
    # lbl1 = Label(window, text = 'Concordance')
    # lbl2 = Label(window, text = 'Rugosité')
    # lbl3 = Label(window, text = 'Defaut')
    # lbl1.grid(column = 1, row = 1,padx=5, pady=5)
    # lbl2.grid(column = 1, row = 2,padx=5, pady=5)
    # lbl3.grid(column = 1, row = 3,padx=5, pady=5)
    #
    # lbR = Label(window, text = 'Rouge')
    # lbG = Label(window, text = 'Vert')
    # lbB = Label(window, text = 'Bleu')
    # lbR.grid(column = 2, row = 0)
    # lbG.grid(column = 3, row = 0)
    # lbB.grid(column = 4, row = 0)
    #
    # varR = StringVar()
    # varG = StringVar()
    # varB = StringVar()
    #
    # rad1 = Radiobutton(window, value='concordance', variable=varR)
    # rad2 = Radiobutton(window, value='roughness', variable=varR)
    # rad3 = Radiobutton(window, value='defaut', variable=varR)
    # rad4 = Radiobutton(window, value='concordance', variable=varG)
    # rad5 = Radiobutton(window, value='roughness', variable=varG)
    # rad6 = Radiobutton(window, value='defaut', variable=varG)
    # rad7 = Radiobutton(window, value='concordance', variable=varB)
    # rad8 = Radiobutton(window, value='roughness', variable=varB)
    # rad9 = Radiobutton(window, value='defaut', variable=varB)
    #
    # rad1.grid(column=2, row=1)
    # rad2.grid(column=2, row=2)
    # rad3.grid(column=2, row=3)
    # rad4.grid(column=3, row=1)
    # rad5.grid(column=3, row=2)
    # rad6.grid(column=3, row=3)
    # rad7.grid(column=4, row=1)
    # rad8.grid(column=4, row=2)
    # rad9.grid(column=4, row=3)
    #
    #
    #
    # lbl = Label(window, text="Niveau 1",foreground = 'black', background='white')
    # lbl.grid(column=0, row=5)
    #
    # l1 = Label(window, text = 'Concordance 3')
    # l2 = Label(window, text = 'Concordance Totale')
    # l3 = Label(window, text = 'Harmonicité')
    # l4 = Label(window, text = 'Tension')
    # l5 = Label(window, text = 'Defaut')
    # l1.grid(column = 1, row = 6,padx=5, pady=5)
    # l2.grid(column = 1, row = 7,padx=5, pady=5)
    # l3.grid(column = 1, row = 8,padx=5, pady=5)
    # l4.grid(column = 1, row = 9,padx=5, pady=5)
    # l5.grid(column = 1, row = 10,padx=5, pady=5)
    #
    # lR = Label(window, text = 'Rouge')
    # lG = Label(window, text = 'Vert')
    # lB = Label(window, text = 'Bleu')
    # lR.grid(column = 2, row = 5)
    # lG.grid(column = 3, row = 5)
    # lB.grid(column = 4, row = 5)
    #
    # vR = StringVar()
    # vG = StringVar()
    # vB = StringVar()
    #
    # r1 = Radiobutton(window, value='concordanceOrdre3', variable=vR)
    # r2 = Radiobutton(window, value='concordanceTotale', variable=vR)
    # r3 = Radiobutton(window, value='harmonicity', variable=vR)
    # r4 = Radiobutton(window, value='tension', variable=vR)
    # r5 = Radiobutton(window, value='defaut', variable=vR)
    # r6 = Radiobutton(window, value='concordanceOrdre3', variable=vG)
    # r7 = Radiobutton(window, value='concordanceTotale', variable=vG)
    # r8 = Radiobutton(window, value='harmonicity', variable=vG)
    # r9 = Radiobutton(window, value='tension', variable=vG)
    # r10 = Radiobutton(window, value='defaut', variable=vG)
    # r11 = Radiobutton(window, value='concordanceOrdre3', variable=vB)
    # r12 = Radiobutton(window, value='concordanceTotale', variable=vB)
    # r13 = Radiobutton(window, value='harmonicity', variable=vB)
    # r14 = Radiobutton(window, value='tension', variable=vB)
    # r15 = Radiobutton(window, value='defaut', variable=vB)
    #
    # r1.grid(column=2, row=6)
    # r2.grid(column=2, row=7)
    # r3.grid(column=2, row=8)
    # r4.grid(column=2, row=9)
    # r5.grid(column=2, row=10)
    # r6.grid(column=3, row=6)
    # r7.grid(column=3, row=7)
    # r8.grid(column=3, row=8)
    # r9.grid(column=3, row=9)
    # r10.grid(column=3, row=10)
    # r11.grid(column=4, row=6)
    # r12.grid(column=4, row=7)
    # r13.grid(column=4, row=8)
    # r14.grid(column=4, row=9)
    # r15.grid(column=4, row=10)
    #
    #







#
# from tkinter import *
# class Interface(Frame):
#
#     """Notre fenêtre principale.
#     Tous les widgets sont stockés comme attributs de cette fenêtre."""
#
#     def __init__(self, fenetre, **kwargs):
#         Frame.__init__(self, fenetre, width=2000, height=1000,borderwidth=1, **kwargs)
#         self.pack()
#         self.nb_clic = 0
#
#         # Création de nos widgets
#         self.messageR1 = Label(self, text="Rouge:")
#         self.messageR1.pack()
#
#         self.var_R1 = StringVar()
#
#         self.concordanceR1 = Radiobutton(fenetre, text="Concordance", variable=self.var_R1, value="concordance")
#         self.roughnessR1 = Radiobutton(fenetre, text="Rugosité", variable=self.var_R1, value="roughness")
#         self.nullR1 = Radiobutton(fenetre, text="Null", variable=self.var_R1, value="null")
#
#         self.concordanceR1.pack()
#         self.roughnessR1.pack()
#         self.nullR1.pack()
#
#         self.messageG1 = Label(self, text="Vert:")
#         self.messageG1.pack()
#
#         self.var_G1 = StringVar()
#
#         self.concordanceG1 = Radiobutton(fenetre, text="Concordance", variable=self.var_G1, value="concordance")
#         self.roughnessG1 = Radiobutton(fenetre, text="Rugosité", variable=self.var_G1, value="roughness")
#         self.nullG1 = Radiobutton(fenetre, text="Null", variable=self.var_G1, value="null")
#
#         self.concordanceG1.pack()
#         self.roughnessG1.pack()
#         self.nullG1.pack()
#
#         self.messageB1 = Label(self, text="Bleu:")
#         self.messageB1.pack()
#
#         self.var_B1 = StringVar()
#
#         self.concordanceB1 = Radiobutton(fenetre, text="Concordance", variable=self.var_B1, value="concordance")
#         self.roughnessB1 = Radiobutton(fenetre, text="Rugosité", variable=self.var_B1, value="roughness")
#         self.nullB1 = Radiobutton(fenetre, text="Null", variable=self.var_B1, value="null")
#
#         self.concordanceB1.pack()
#         self.roughnessB1.pack()
#         self.nullB1.pack()
#

# space = ['concordance', 'roughness']
#
# window = Tk()
# window.title("Paramètres")
# window.geometry('500x300')
# interface = Interface(window, space, 0)
# interface.mainloop()
# interface.destroy()

##################### CONSTRUCTION DICTIONNAIRE ##############################
# with open('dic_concordance_normTot.pkl', 'rb') as f:
#     dic_concordance = pickle.load(f)
#
# with open('dic_roughness_normTot.pkl', 'rb') as f:
#     dic_roughness = pickle.load(f)
#
# with open('dic_concordanceOrdre3.pkl', 'rb') as f:
#     dic_concordanceOrdre3 = pickle.load(f)
#
# with open('dic_concordanceTotale.pkl', 'rb') as f:
#     dic_concordanceTotale = pickle.load(f)
#
# with open('dic_harmonicity.pkl', 'rb') as f:
#     dic_harmonicity = pickle.load(f)
#
# # with open('dic_tension.pkl', 'rb') as f:
# #     dic_tension = pickle.load(f)
#
# space = ['concordance', 'roughness', 'concordanceOrdre3', 'concordanceTotale']
# Dic_Harm = {}
# K0, decr0, σ0 = 7,1/2,0.01
# Dic_Harm[(K0, decr0, σ0)] = {'K': K0, 'decr': decr0, 'σ': σ0}
#
# Dic_Harm[(K0, decr0, σ0)]['concordance'] = dic_concordance
# Dic_Harm[(K0, decr0, σ0)]['roughness'] = dic_roughness
# Dic_Harm[(K0, decr0, σ0)]['concordanceOrdre3'] = dic_concordanceOrdre3
# Dic_Harm[(K0, decr0, σ0)]['concordanceTotale'] = dic_concordanceTotale
# Dic_Harm[(K0, decr0, σ0)]['harmonicity'] = dic_harmonicity
#
# with open('Dic_Harm.pkl', 'wb') as f:
#     pickle.dump(Dic_Harm, f)
############################################################



# with open('dic_concordanceOrdre3.pkl', 'rb') as f:
#     dic_concordanceOrdre3 = pickle.load(f)
#
# with open('dic_concordanceTotale.pkl', 'rb') as f:
#     dic_concordanceTotale = pickle.load(f)
#
# print(len(dic_concordanceTotale))
# print(dic_concordanceTotale)

# with open('Dic_id.pkl', 'rb') as f:
#     Dic_id = pickle.load(f)
#
# dic_concordanceOrdre3 = {id: dic_concordanceOrdre3[Dic_id[id]+1] for id in Dic_id}
# dic_concordanceTotale = {id: dic_concordanceTotale[Dic_id[id]+1] for id in Dic_id}
#
# with open('dic_concordanceOrdre3.pkl', 'wb') as f:
#     pickle.dump(dic_concordanceOrdre3, f)
#
# with open('dic_concordanceTotale.pkl', 'wb') as f:
#     pickle.dump(dic_concordanceTotale, f)

# Dic_id = {}
# i = 0
#
# for cle in Dic_iv:
#     if isinstance(cle, str):
#         Dic_id[cle] = i
#         print(cle + ': {}'.format(i))
#         i += 1
#
#
#
# with open('Dic_id.pkl', 'wb') as f:
#     pickle.dump(Dic_id, f)
