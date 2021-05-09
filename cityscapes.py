import cv2
import numpy as np
from sklearn.decomposition import PCA
import os

list = os.listdir('C:/Users/Jainy/Downloads/cityscapes_data/cityscapes_data/train')
 # dir is your directory path
total_images = len(list)
#print (number_files)
#s = 'C:/Users\Jainy\Desktop\cityscapes_data\train\' + '1' + '.jpg'
for i in range(1, total_images+1):
    img = cv2.imread('C:/Users/Jainy/Desktop/cityscapes_data/train/' + str(i) + '.jpg')

    crop = img[0:128, 0:256]

    blue,green,red = cv2.split(crop)

    #initialize PCA with first 20 principal components
    pca = PCA(50)

    #Applying to red channel and then applying inverse transform to transformed array.
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    #Applying to Green channel and then applying inverse transform to transformed array.
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    #Applying to Blue channel and then applying inverse transform to transformed array.
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)

    img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)

    #viewing the compressed image
    cv2.imshow('crop_lower', img_compressed)

    #cv2.imshow('kjd', red)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
