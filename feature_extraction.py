import cv2
import numpy as np
import pandas as pd 
from skimage import io, color, img_as_ubyte
import os
from tqdm.auto import tqdm
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    return contrast

def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return energy

def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return correlation
	
	
def feature_extraction(input_file):
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    scale_percent=0.50
    width = int(img.shape[1]*scale_percent)
    height = int(img.shape[0]*scale_percent)
    dimension =(width,height)
    resized = cv2.resize(img,dimension)
    print('Original Shape:', img.shape)
    histogram = np.mean(cv2.calcHist([resized],[0],None,[256],[0,256]))
    sobelX = np.mean(cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=5))
    sobelY = np.mean(cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=5))
    image = img_as_ubyte(resized)
    print('Resize Shape:', np.shape(image))
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bin
    inds = np.digitize(image, bins)
    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
    contrast = np.mean(contrast_feature(matrix_coocurrence))
    correlation = np.mean(correlation_feature(matrix_coocurrence))
    energy = np.mean(energy_feature(matrix_coocurrence))
    homogeneity = np.mean(homogeneity_feature(matrix_coocurrence))
    return histogram, sobelX, sobelY, contrast, correlation, energy, homogeneity

