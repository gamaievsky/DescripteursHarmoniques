from music21 import *
from music21.converter.subConverters import ConverterMusicXML
from PIL import Image
import os

conv_musicxml = ConverterMusicXML()
stream1 = stream.Stream()
l = [0,2.5,7.5,9]
for p in l:
    n = note.Note()
    n.pitch = pitch.Pitch(p)
    if not isinstance(p,int):
        n.pitch.convertMicrotonesToQuarterTones(inPlace=True)
    n.quarterLength = 4.0
    print(n.pitch.alter)
    stream1.append(n)
for i in range(len(stream1)):
    stream1[i].offset = 0.0

l_mod = [(l+[12])[i+1] - (l+[12])[i] for i in range(len(l))]
stream1[0].lyric = str(l_mod)

title = 'quarterTone_brouillon'
filepath = '/Users/manuel/Dropbox (TMG)/Thèse/Estrada/'
out_filepath = conv_musicxml.write(stream1, 'musicxml', fp=filepath + title + '.XML', subformats=['png'])
im = Image.open(out_filepath)





# width, height = im.size
# region = im.crop((130, 0, 610, height))
# region.save(out_filepath[:-6] + '.png')

# os.remove(filepath + title + '.XML')
# os.remove(filepath + title + '-1.png')


##################### AFFICHER UN ACCORD ##########################
# conv_musicxml = ConverterMusicXML()
# stream1 = stream.Stream()
# l = [0,1,2,3,4,5,6,7,8]
# l_mod = [(l+[12])[i+1] - (l+[12])[i] for i in range(len(l))]
# c = chord.Chord(l)
# c.quarterLength = 4.0
# c.lyric = str(str(l_mod))
# stream1.append(c)
#
# title = 'acc71-1_brouillon'
# filepath = '/Users/manuel/Dropbox (TMG)/Thèse/Estrada/'
# out_filepath = conv_musicxml.write(stream1, 'musicxml', fp=filepath + title + '.XML', subformats=['png'])
# im = Image.open(out_filepath)
# width, height = im.size
# region = im.crop((130, 0, 610, height))
# region.save(out_filepath[:-6] + '.png')
#
# os.remove(filepath + title + '.XML')
# os.remove(filepath + title + '-1.png')
####################################################################
