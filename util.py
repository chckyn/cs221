import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class Image(object):
    def __init__(self, data, patches, label):
        self.label = label
        self.patches = patches
        self.__raw_data = data

    def view(self):
        fig = plt.imshow(self.__raw_data)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    def __repr__(self):
        return 'Image(%r, %r, %r)'\
                % (self.__raw_data, self.patches, self.label)


