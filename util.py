import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import const

IMG_HEIGHT = const.IMG_HEIGHT
IMG_WIDTH = const.IMG_WIDTH

class Image(object):
    def __init__(self, pixels, label):
        self.label = label
        self.pixels = pixels

    def view(self):
        fig = plt.imshow(self.pixels.reshape(IMG_HEIGHT,IMG_WIDTH))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    def __repr__(self):
        return 'Image(%r, %r)'\
                % (self.pixels, self.label)


