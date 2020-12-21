# Coïncidence de partiels

from p5 import *
import pickle
import numpy as np

def setup():
    global tone1, tone2, tone3
    global f0
    f0 = create_font("Arial.ttf", 16)
    size(900, 400)
    tone1 = Tone('Do',261)
    # tone2 = Tone('Ré',292)
    tone2 = Tone('Mi',329)
    tone3 = Tone('Sol♯', 415)

def draw():
    background(255)
    stroke(0,0,255,75)
    stroke_weight(2)
    for f in list(set(tone1.harm) & set(tone2.harm) & set(tone3.harm)):
        line((70 + f*100/260.0,50), (70 + f*100/260.0,350))
    for f in list(set(tone1.harm) & set(tone2.harm)):
        line((70 + f*100/260.0,75), (70 + f*100/260.0,225))
    for f in list(set(tone1.harm) & set(tone3.harm)):
        line((70 + f*100/260.0,75), (70 + f*100/260.0,325))
    for f in list(set(tone2.harm) & set(tone3.harm)):
        line((70 + f*100/260.0,175), (70 + f*100/260.0,325))
    stroke(0)
    tone1.draw(100)
    tone2.draw(200)
    tone3.draw(300)

def mouse_pressed():
    save('/Users/manuel/Dropbox (TMG)/Thèse/Manuscript/schema.png')
    print('Saved in Manuscript')


class Tone:
    def __init__(self, name, freq):
        self.name = name
        self.freq = freq
        self.n = int(7.5*260/freq)
        self.harm = [(k+1)*self.freq for k in range(self.n)]

    def draw(self, y):
        text_font(f0)
        fill(0)
        text(self.name, (20,y-10))
        translate(70,y)
        stroke_weight(1.5)
        line((0, 0), (750, 0))
        for f in self.harm:
            Y = f*100/260.0
            line((Y,-10), (Y,10))
        translate(-70,-y)


if __name__ == '__main__':
    run()




#######################################

# from p5 import *
#
# click = False
#
# def setup():
#    size(1200, 800)
#    fill(0,102)
#    global img
#    img = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc18.png')
#    img1 = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc18-1.png')
#    img2 = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc18-2.png')
#
# def draw():
#    background(255)
#    image(img,(100,100))
#    stroke(0)
#    no_fill()
#    rect((100, 100), img.width, img.height)
#
#    if click:
#        stroke(0)
#        fill(200,100)
#        rect((400, 50),600, 600)
#
#        R = 200
#        x0 = 400 + 300
#        y0 = 50 + 300
#        for i in range(1,4):
#            image(img, (x0 + R*np.cos(2*np.pi*i / 3), y0 + R*np.sin(2*np.pi*i / 3)), ((img.width/img.height)*45,45))
#            stroke(0)
#            no_fill()
#            rect((x0 + R*np.cos(2*np.pi*i / 3), y0 + R*np.sin(2*np.pi*i / 3)), (img.width/img.height)*45,45)
#
#
#
# def mouse_pressed():
#     global click
#     if (100 < mouse_x < 100 + img.width) and (100 < mouse_y < 100 + img.height):
#         if click: click = False
#         else: click = True
#
#
#
#
# if __name__ == '__main__':
#     run()
# #


#################### DICTIONNAIRE IMAGES ##########################
# with open('Dic_card.pkl', 'rb') as f:
#     Dic_card = pickle.load(f)
# with open('Dic_img.pkl', 'rb') as f:
#     Dic_img = pickle.load(f)
# # Dic_img = {}
#
# def setup():
#         size(320,240)
#         # Make a new instance of a PImage by loading an image file
#         global Dic_img
#         for ind in range(56,78):
#             img = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc{}.png'.format(ind))
#             Dic_img[ind] = img
#             for perm in range(1, Dic_card[ind]+1):
#                 img = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc{}-{}.png'.format(ind,perm))
#                 Dic_img['{}-{}'.format(ind,perm)] = img
#
#         with open('Dic_img.pkl', 'wb') as f:
#             pickle.dump(Dic_img, f)
#
#
#         # img = load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc12.png')
#
# def draw():
#         background(255)
#         # Draw the image to the screen at coordinate (0,0)
#
# if __name__ == '__main__':
#         run()
####################################################################


# def setup():
#         global dic_img
#         dic_img = {}
#         for ind in range(1,78):
#             img = p5.load_image('/Users/manuel/Dropbox (TMG)/Thèse/Estrada/acc{}'.format(ind))
#             dic_img[ind] = img
#
#         with open('dic_img.pkl', 'wb') as f:
#             pickle.dump(dic_img, f)
#
#
# def draw():
#         background(255)
#
#         # stroke(175)
#         # line((width/2,0), (width/2,height))
#
#         text_font(f)
#         fill(0)
#
#         # text_align("CENTER")
#         text('This text is centered.',(width/2,60))
#
#
# if __name__ == '__main__':
#         run()
#
#
# rings = [] # Create the array
# numRings = 50
# currentRing = 0
#
# def setup():
#     size(800, 500)
#
#     for i in range(numRings):
#         rings.append(Ring())
#
# def draw():
#     background(0)
#
#     for r in rings:
#         r.grow()
#         r.display()
#
# def mouse_pressed():
#     global currentRing
#     rings[currentRing].start(mouse_x, mouse_y)
#
#     currentRing += 1
#     if currentRing > numRings:
#         currentRing = 0
#
# class Ring:
#     def __init__(self):
#         self.x = 0
#         self.y = 0
#         self.diameter = 0
#         self.on = False
#
#     def start(self, xpos, ypos):
#         self.x = xpos
#         self.y = ypos
#
#         self.diameter = 1
#         self.on = True
#
#     def grow(self):
#         if self.on:
#             self.diameter += 0.5
#             if self.diameter > 400:
#                 self.on = False
#                 self.diameter = 1
#
#     def display(self):
#         if self.on:
#             no_fill()
#             stroke_weight(4)
#             stroke(204, 153)
#             ellipse((self.x, self.y), self.diameter, self.diameter)
#
# if __name__ == '__main__':
#     run()
