# -*- coding: utf-8 -*-
"""
Utility functions

@author: Maxim Ziatdinov
"""

import tensorflow as tf
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def predict(imgdata, prediction_model):
    '''Applies a trained neural network to an image ot to a stack of images'''
    if len(imgdata.shape) == 2:
        imgdata = np.expand_dims(imgdata, axis=0)
    imgdata = (imgdata - np.amin(imgdata))/np.ptp(imgdata)
    imgdata = np.expand_dims(imgdata, axis=3)
    imgdata = tf.cast(imgdata, tf.float32)
    prediction = prediction_model.predict(imgdata)
    return prediction

def threshold_output(imgdata, t=0.5): #images 2D
    '''Binary threshold of an output of a neural network'''
    imgdata_ = cv2.threshold(imgdata, t, 1, cv2.THRESH_BINARY)[1]
    return imgdata_

def filter_isolated_cells(imgdata, filter='below', th=15): #maybe above, check image
    '''Filters out blobs above cetrain size
    in the thresholded neural network output'''
    label_img, cc_num = ndimage.label(imgdata)
    cc_areas = ndimage.sum(imgdata, label_img, range(cc_num + 1))
    if filter == 'above':
        area_mask = (cc_areas > th)
    else:
        area_mask = (cc_areas < th)
    label_img[area_mask[label_img]] = 0
    label_img[label_img > 0] = 1
    return label_img

def find_blobs(imgdata):
    '''Finds position of defects in the processed output
       of a neural network via center of mass method'''
    labels, nlabels = ndimage.label(imgdata)
    coordinates =  ndimage.center_of_mass(
        imgdata, labels, np.arange(nlabels)+1)
    coordinates = np.array(coordinates, dtype=np.float)
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates

def get_contours(thresh_image, t_area=5):
    '''Returns major axis of each contour and the corresponding coordinate'''
    thresh_image = cv2.convertScaleAbs(thresh_image)
    contours_o = cv2.findContours(
        thresh_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_o[0] if len(contours_o) == 2 else contours_o[1]
    area_all = []
    ma_all = []
    cxy_all = np.empty((0, 2))
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > t_area:
            area_all.append(area)
            (y, x), (m, M), _ = cv2.fitEllipse(cnt)
            ma_all.append(M)
            xy_i = np.array([x, y]).reshape(1, -1)
            cxy_all = np.append(cxy_all, xy_i, axis=0)
    ma_all = np.array(ma_all)
    area_all = np.array(area_all)
    return area_all, ma_all, cxy_all

def inference(imgdata, model, thresh=0.5, thresh_blobs=15):
    '''Obtain position of defects from the neural network output for a single image'''
    decoded_img = predict(imgdata, model)
    decoded_img = threshold_output(decoded_img[0, :, :, 0], thresh)
    decoded_img = filter_isolated_cells(decoded_img, thresh_blobs)
    defcoord = find_blobs(decoded_img)
    return defcoord


def draw_boxes(imgdata, defcoord, bbox=16, figsize_=(6, 6)):
    '''Draws boxes cetered around the extracted dedects'''
    fig, ax = plt.subplots(1, 1, figsize=figsize_)
    ax.imshow(imgdata, cmap='gray')
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        starty = int(round(point[1] - bbox))
        p = patches.Rectangle(
            (starty, startx), bbox*2, bbox*2,
            fill=False, edgecolor='red', lw=2)
        ax.add_patch(p)
    ax.grid(False)
    plt.show()
    return fig
