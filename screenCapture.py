# -*- coding: utf-8 -*-

from PIL import ImageGrab
import numpy

def scr_cap():
    imbuf = ImageGrab.grab()
    im=numpy.array(imbuf)
    return im

if __name__ == "__main__":
    im = ImageGrab.grab()
    im.show()