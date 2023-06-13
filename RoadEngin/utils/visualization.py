import os
import pandas as pd
from google.colab import files
import json
import xmltodict
import csv
import numbers
from shutil import copyfile
from treelib import Node, Tree
from time import sleep
from tqdm.auto import tqdm
import glob
import random
import cv2
import numpy as np
from time import time
import gdown
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

from out_manipulation import out_resize, out_offset


def img_out(img_path, out_path, plot=True):
    """
    plot in parallel the image and the out
    :param plot: if set to True a parallel plot of img and out will be plot
    :param img_path: usually gcs path or gdrive path of the original image
    :param out_path: numpy array resulted from the detections.
    :return: image and out arrays
    """
    img = cv2.imread(img_path)
    shape = img.shape
    out = out_resize(np.load(out_path), shape)

    if plot:
        plt.rcParams['figure.figsize'] = 7 * 2, int(7 * shape[0] / shape[1] * 1)

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(out)
        plt.axis("off")

        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        plt.show()

    return img, out


def plot_detections(outs_dict, num_cols=4):
    """
    Plot an array of images based on outs_dict binary data.
    """
    n_img = len(outs_dict.items())
    num_cols = min(num_cols, n_img)
    shape = img.shape
    num_rows = int(n_img / num_cols) + (n_img % num_cols > 0)
    plt.rcParams['figure.figsize'] = 7 * num_cols, int(7 * shape[0] / shape[1] * num_rows)
    i = 1
    for damage, out_dict in outs_dict.items():
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(out_dict)
        plt.title(damage)
        plt.axis("off")
        i += 1
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.show()



