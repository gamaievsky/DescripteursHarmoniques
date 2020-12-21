
# SPECTRE
mode = 'auto' #manual
noteDeReferencePourLeTunning = "C4"

# partiels = [1,2,3,4,5,6,7,8,9,10,11]
# amplitudes = [1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 1/11]

# Spectre
timbre_def = (11,1/2,0.005)
shepard = False

#PARAMETRES DES DESCRIPTEURS
#Dissonance
β1 = 3.5
β2 = 5.75
P_ref = 20*(10**(-6))
#Tension:
δ = 0.6

## NORMALISATIONS
type_diss = 'produit' #'produit', 'minimum'
norm_diss = True
norm_crossConc = 'energy' #'energy', 'energy + conc', 'first' ##energy : même normalisation que dans le calcul de la consonance
norm_crossConcTot = 'energy' #'energy', 'energy + conc', 'first' ##energy : même normalisation que dans le calcul de la consonance totale
type_harmChange = 'absolute' # 'absolute', 'relative'
norm_harmChange = 'general' # 'None', 'frame_by_frame', 'general'
norm_diffConc = 'energy' # 'energy', 'energy + conc', 'first'
norm_harm = 2 # La puissance dans le calcul de l'harmonicité. 1 : amplitude, 2 : énergie
norm_diffConcContext = 'energy'
type_diffDiss = 'produit'
type_harmNov = 'dyn' #'dyn', 'stat'

## CONTEXTE ET MEMOIRE
memory_size = 2 # "full", int # entier n>=1, auquel cas la mémoire ne dure que n+1 accords
memory_type = 'mean' #'max','mean'
memory_decr_ponderation = 1
norm_Novelty = 'energy' # 'None', 'energy'

# AFFICHAGE
plot_score = False
plot_class = False
plot_descr = True
plot_abstr = False

aff_score = True
link = True
color_abstr = 'b'

## COMPARAISON DE TIMBRES ET DES DESCRIPTEURS
one_track = True
compare_instruments = False
compare_scores = False
compare_descriptors = False

type_Temporal = 'differential' #'static', 'differential'
type_Normalisation = 'by timbre' #'by curve', 'by timbre'

visualize_trajectories = False
visualize_time_grouping = True

correlation = False
pca = False
