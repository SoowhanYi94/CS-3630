#!/usr/bin/env python

##############
#### Your name: Soowhan Yi
##############

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color, measure
from skimage.measure import LineModelND, ransac
# import matplotlib.pyplot as plt
# from PyQt5 import QtCore
import ransac_score
import joblib


class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        ########################
        feature_data = []
        for pixel in data:
            temp_pixel = pixel
            temp_pixel = exposure.rescale_intensity(temp_pixel)
            #temp_pixel = exposure.equalize_adapthist(temp_pixel)
            img_feature = feature.hog(temp_pixel, orientations=10, pixels_per_cell=(24,24), cells_per_block=(6,6), block_norm='L2-Hys', visualize=False, feature_vector=True, multichannel=True)
            feature_data.append(img_feature)

        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        
        ########################
        ######## YOUR CODE HERE
        ########################
        self.classifier = svm.SVC(kernel = 'linear').fit(train_data, train_labels)
        joblib.dump(self.classifier, 'trained_model.pkl')


    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ######## YOUR CODE HERE
        ########################
        predicted_labels = self.classifier.predict(data)
        # Please do not modify the return type below
        return predicted_labels

    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        ######## YOUR CODE HERE
        ########################
        model = LineModelND()
        model.estimate(data)
        slope = []
        intercept = []
        for image in data:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            gray_image = np. dot(image[...,: 3], rgb_weights)  
            edges = feature.canny(gray_image, sigma=2, low_threshold=0.9, high_threshold=0.999, mask=None, use_quantiles=True)
            # v = viewer.ImageViewer(edges)
            # v.show()
            ransac_model, inliers = ransac(np.argwhere(edges), model_class = measure.LineModelND, min_samples = 2, residual_threshold = 1, max_trials=15, stop_sample_num= float('inf'), stop_residuals_sum= 0, stop_probability=1, random_state= 1, initial_inliers=None)
            outliers = inliers == False
            (origin, direction) = np.round(ransac_model.params, 5)
            slope_temp = direction[0] / direction[1] #(finding change in y over change in x)
            intercept_temp = origin[0] - slope_temp * origin[1] #(finding y intercept)
            # plt.imshow(gray_image, cmap=plt.cm.gray)
            # line_y = np.arange(0, 250)
            # line_x = ransac_model.predict_x(line_y)
            # p1 = origin
            # p2 = direction
            # plt.plot(line_y, line_x, color='#ff0000', linewidth=1.5)
            # plt.show()
            slope.append(slope_temp)
            intercept.append(intercept_temp)
        # Please do not modify the return type below

        return slope, intercept

# def plot_ransac(img, p1, p2):
#     plt.imshow(img, cmap=plt.cm.gray)
#     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#ff0000', linewidth=1.5)
#     plt.show()

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))
    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")
    print(s)
    print("\n")
    print(i)


if __name__ == "__main__":
    main()
