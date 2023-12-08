import os
import pandas as pd
import json
import csv
import numbers
from shutil import copyfile
from time import sleep
import glob
import random
import cv2
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
from constants import *


def out_offset(out, offset=True):
    """
    In the case of Swin, detections has an offset according to the CONV_LU dictionary, therefore it should be moved 1
    unit.
    :param out: array
    :param offset: True: if out has an offset, integer: if the offset is different from 1
    :return: a fixed out.
    """
    if offset:
        return out + offset
    else:
        return out


def out_resize(out, shape):
    """
    In some cases out has not the shape of the corresponding image, so it requires to change it to deal with other
    functionalities.
    :param out: array
    :param shape: referenced image shape (img.shape)
    :return: a resized out array
    """
    if out.shape != shape[:2]:
        return cv2.resize(out, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        return out


def get_contours(img, kernelSize=3):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    ret, thresh = cv2.threshold(img.astype(np.uint8), 0.3, 1.01, 0)
    img_dilated = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def get_out_dict(out, layer_list=CRACKS, kernel_size=20, iterations=1):
    """
    Separate the layers listed according to the engin tag, and dilates the detections according to kernel_size and
    iterations.
    :param out: ID detections array.
    :param layer_list: list of tags to get the layers dict.
    :param kernel_size: the size in pixels of the kernel to dilate.
    :param iterations: number of dilating that will be performed.
    :return: a dictionary of out layers according to layer_list and out detections.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    outs_dict = {}
    for ID in np.unique(out):
        if CONV_LU_INV[ID] in layer_list:
            dmg_out = np.array(out == ID, dtype=np.uint8)
            dilated = cv2.dilate(dmg_out, kernel, iterations=iterations)
            outs_dict[CONV_LU_INV[ID]] = dilated
    return outs_dict


def clean_outs_dict(outs_dict: dict, min_area=10):
    return {damage: out_dict for damage, out_dict in outs_dict.items() if np.sum(out_dict != 0) > min_area}


def get_dmg_list(all_dmg, key='FR_'):
    return [dmg for dmg in all_dmg if key in dmg]


def unique(elements):
    u = []
    [u.append(el) for el in elements if el not in u]
    return u


def get_stencil(out_comb, cont):
    stencil = np.zeros(out_comb.shape, dtype=np.uint8)
    stencil = cv2.fillPoly(stencil, [cont], 1)
    return out_comb * stencil


def update_cont(outs_dict, cont, d1, d0):
    outs_dict[d1] = cv2.fillPoly(outs_dict[d1], pts=[cont], color=1)
    outs_dict[d0] = cv2.fillPoly(outs_dict[d0], pts=[cont], color=0)
    return outs_dict


# Cracks general analysis
def cracks_out(out):
    """
    Preliminary analysis of crack detections. Linear and area crack analysis based on biggest area of a contoured set of
    detections. If the biggest one has more than 50% the region area, this will take the detection. If an FR is the
    biggest one, a hierarchy analysis is performed and the distress that is at least 30% of the FR area will take the
    detection.
    :param out: detections array.
    :return: an updated version of out.
    """
    for crack_ids in [AREA_CRACKS_ID, LINE_CRACKS_ID]:
        out_cracks = np.array(np.isin(out, crack_ids), np.uint8)
        conts = get_contours(out_cracks, int(out.shape[0] / 100))
        for cont in conts:
            stencil = get_stencil(out_cracks * out, cont)
            IDs = np.unique(stencil)
            if len(IDs) > 1:
                areas = [np.sum(stencil == ID) for ID in IDs]
            total_area = np.sum(areas)
            if np.max(areas) > .5 * total_area:
                # analysis
                df = pd.DataFrame(dict(zip(IDs, areas)), columns=['ID', 'area'])
                df.sort_values('area', ignore_index=True, inplace=True)
                df['label'] = [CONV_LU[ID] for ID in df.ID]
                if 'FR' not in df.label[0]:
                    out[stencil != 0] = df.ID[0]
                else:
                    hierarchy = FR_DICT[df.label[0]]


def cracks_FR_analysis(crack_outs: dict, FR_analysis: dict):
    """
    Perform a first analysis over FR detections based on a hierarchy per FR classification. The first
    :param crack_outs: dictionary with crack layers. Most of the cases FB has some parts detected as FR, and it is
    expected that future versions will change this step. If there is no other cracks nearby the FR, FR will be analyzed
    depending on the methodology.
    :param FR_analysis: {FR_: [hierarchy]}, depending on the project PC or FB domains the reports, so this could
    be edited here.
    :return: an update of crack_outs.
    """
    cracks = crack_outs.keys()
    out_FRs = get_dmg_list(cracks, 'FR_')
    for FR in out_FRs:
        if FR in FR in FR_analysis.keys():
            analysis = unique(FR_analysis[FR] + list(cracks))
            for dmg in analysis:
                if dmg in cracks:
                    out_comb = crack_outs[dmg] + crack_outs[FR]
                    for cont in get_contours(np.array(out_comb != 0)):
                        out_temp = get_stencil(out_comb, cont)
                        if len(np.unique(out_temp)) > 2:
                            crack_outs = update_cont(crack_outs, cont, dmg, FR)
    return clean_outs_dict(crack_outs)


def cracks_merge(crack_outs, dmg1, dmg2):
    """
    Analyze particular cracks arranged or listed hierarchically in a dictionary.
    :param crack_outs: dictionary with crack layers.
    :param hirarchy_dict: dictionary of cracks to analyze.
    :return: an update of crack_outs.
    """

    return clean_outs_dict(crack_outs)
# def cracks_cleanup(crack_outs,):
