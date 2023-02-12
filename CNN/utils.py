import numpy as np
import uuid
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from keras import backend as K
import ntpath
from sklearn.utils import shuffle
import json


from balance import image_aug_balance

import os

import numpy as np

from skimage import data, img_as_float, img_as_ubyte
from skimage.segmentation import chan_vese
from skimage.color import rgb2gray
from skimage.io import imsave, imread
from skimage.transform import resize

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root=None, data_per_name=5):
    
    all_files = [os.path.join(root,i) for i in os.listdir(root) if dataset["name"] in i]
    all_files.sort()

    folds = []

    for i in range(0, len(all_files), data_per_name):

        curr_fold_files = all_files[i:i+data_per_name]

        train_x_f = [j for j in curr_fold_files if "-train-X.npy" in j][0]
        train_y_f = [j for j in curr_fold_files if "-train-Y.npy" in j][0]

        test_x_f = [j for j in curr_fold_files if "-test-X.npy" in j][0]
        test_y_f = [j for j in curr_fold_files if "-test-Y.npy" in j][0]

        train_extra_x_f = [j for j in curr_fold_files if "-train-X-aug.npy" in j][0]

        folds.append((len(folds), train_x_f, train_y_f, test_x_f, test_y_f, train_extra_x_f))

    return folds


class SGDReduceScheduler:

    def __init__(self, model, rate=0.2, epochs=10):
        self.model = model
        self.epochs = epochs
        self.rate = rate
        self.count = 0.

    def on_epoch_end(self):
        self.count += 1
        if self.count//self.epochs == self.count/self.epochs:
            K.set_value(
                self.model.optimizer.lr, 
                K.get_value(self.model.optimizer.lr) * self.rate
            )
            print('>> new learning rate', K.get_value(self.model.optimizer.lr))
