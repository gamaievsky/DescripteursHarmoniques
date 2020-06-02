from p5 import *

def setup():
    size(400, 300)

def draw():
    background(255)
    stroke_weight(5)
    stroke(0)
    fill(150)
    rect((50, 50), 200, 200)

if __name__ == '__main__':
    run()
