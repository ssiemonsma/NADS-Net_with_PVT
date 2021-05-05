# import all the necessary libraries
import os
import argparse
from keras import Model
import cv2
import math
import time
import numpy as np
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from poseNetF import get_testing_model, get_training_model
from keras.layers import Activation, Input, Dense, Conv2D, concatenate, BatchNormalization, ReLU, Flatten
from keras.layers.merge import Concatenate
from keras.utils import plot_model
import tkinter
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
from PIL import Image
import matplotlib
import pylab as plt
import tensorflow as tf
from GUI import GUI
from google_drive_downloader import GoogleDriveDownloader as gd

#Body joint pairs for Part Affinity Field
limbSeq = [[4,3],[3,2],[2,1],[4,5],[5,6],[6,7],[4,8],[4,9]]
# x and y direction of part affinity field paired
mapIdx = [ [0,1],[2,3],[4,5],[6,7],[8,9],[10,11], [12,13], [14,15]]

#This code is to add padding to the input image if the image height and weight is not a multiple of 32 or (2^5)
def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

#Preprocess the input image and generate heatmaps from the Deep model
def process(input_image, params, model_params):

    k_scaler = 1
    oriImg = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    #Scale for resizing the input image
    fx1 = (288/oriImg.shape[0])
    fy1 = (288/oriImg.shape[1])

    imageToTest = cv2.resize(oriImg, (0, 0), fx=fy1, fy=fy1, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    #Normalize image same as ResNet-50 normalization since we are transfer learning ResNet-50
    input_img[:, :, 0] -= 103.939
    input_img[:, :, 1] -= 116.779
    input_img[:, :, 2] -= 123.68

    start = time.time()
    #Insert input image and predict
    output_blobs = model.predict(input_img)
    print ("prediction", time.time() - start)
    oriImg = cv2.resize(imageToTest, (0, 0), fx=1/k_scaler, fy=1/k_scaler, interpolation=cv2.INTER_CUBIC)

    #resize the heatmap and PAF to original size
    heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv2.resize(heatmap, (imageToTest.shape[1], imageToTest.shape[0]), interpolation=cv2.INTER_CUBIC)

    paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
    paf = cv2.resize(paf, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    paf = cv2.resize(paf, (imageToTest.shape[1], imageToTest.shape[0]), interpolation=cv2.INTER_CUBIC)

    all_peaks = []
    peak_counter = 0

    start = time.time()

    #Non-maximum suppression to find the peak point from heatmaps
    map_Gau = cv2.GaussianBlur(heatmap,(5,5),2)
    for part in range(9):
        map_ori = heatmap[:, :, part]
        map = map_Gau[:, :, part]
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    print ("1", time.time() - start)
    start = time.time()

### Process part affinity field and find true connection between body joints
    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(8):
        score_mid = paf[:, :, [x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))


                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \

                        for I in range(len(startend))])

                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 11))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    print ("2", time.time() - start)
    start = time.time()
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 7:
                    row = -1 * np.ones(11)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur (any subset that has less than 4 components)
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    print ("3", time.time() - start)

    start = time.time()

    canvas = cv2.resize(input_image, (0, 0), fx=fy1, fy=fy1, interpolation=cv2.INTER_CUBIC)

    #Process Seatbelt Segmentation
    # Uncomment this section if you don't want to show Seatbelt segmentation

    seatbelt = np.squeeze(output_blobs[2])
    seatbelt = seatbelt[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3]]
    seatbelt = cv2.resize(seatbelt, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_CUBIC)

    thres = 0.01
    seatbelt[seatbelt>thres] = 255
    seatbelt[seatbelt<=thres] = 0
    seatbelt= seatbelt.astype(int)
    p = (seatbelt == 0)

    canvas[:,:,0] = np.where(p, canvas[:,:,0], 0)
    canvas[:,:,1] = np.where(p, canvas[:,:,1], 0)
    canvas[:,:,2] = np.where(p, canvas[:,:,2], 255)

# Visualize the detected body joints and skeletons
    keypoints=[]
    for s in subset:
        keypoint_indexes = s[:9]
        person_keypoint_coordinates = []
        for index in keypoint_indexes:
            if index == -1:
                # "No candidate for keypoint"
                X, Y = 0, 0
            else:
                X, Y = candidate[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        keypoints.append((person_keypoint_coordinates, 1 - 1.0 / s[9]))

    kp_sks = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[3,9],[5,8],[8,9]]
    kp_sks = np.array(kp_sks)-1
    clrs = (124,252,0)
    for ind, key in enumerate(keypoints):
        keys = []
        for i in key[0]:
            keys.append(i[0])
            keys.append(i[1])
        x = np.array(keys[0::2])
        y = np.array(keys[1::2])

        for sk in kp_sks:
            if np.all(x[sk]>0):
                cv2.line(canvas, (int(x[sk[0]]*k_scaler),int(y[sk[0]]*k_scaler)), (int(x[sk[1]]*k_scaler),int(y[sk[1]]*k_scaler)), clrs, 3)
        for k in range(9):
            cv2.circle(canvas,((int(x[k]*k_scaler)),(int(y[k]*k_scaler))) , 3, (0,128,0), thickness=-1)
    print(canvas.shape)

    return canvas


if __name__ == '__main__':
    model = get_testing_model()

    # Get the weight file
    keras_weights_file = "Final_weights.h5"
    gd.download_file_from_google_drive(file_id='1_E4oN8DdPSEbsYcwwGlwTiDAGYl279Pe',
                                       dest_path='./' + keras_weights_file)
    model.load_weights(keras_weights_file, by_name=True)
    params, model_params = config_reader()


    class Pose(GUI):
        def __init__(self, window, window_title, video_source):
            GUI.__init__(self, window, window_title, video_source)

        def process(self, frame):
            start = time.time()
            result = process(frame, params, model_params)

            print ("process fps: ", 1.0/(time.time()-start))
            tt = 1.0/(time.time()-start)
            return result, tt

        def open(self):
            inputFileName = filedialog.askopenfilename()
            self.window.destroy()
            Pose(tkinter.Tk(), "Tkinter and OpenCV", inputFileName)

        def update(self):
            ret, frame = self.vid.get_frame()
            tt = 30
            if ret:
                if self.value == 0:
                    out_frame, tt = self.No_process(frame)
                elif self.value == 1:
                    out_frame, tt = self.process(frame)
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(out_frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                self.box.delete(1.0,'end')
                self.box.insert('insert', str(round(tt)))
            self.window.after(self.delay, self.update)

    #Change the input video file if necessary.
    Pose(tkinter.Tk(), "Tkinter and OpenCV", 'inputs/test1.mp4')
    #comment the above line and uncomment line of code below to run the file for Webcam
    # Pose(tkinter.Tk(), "Tkinter and OpenCV", 0) ##For webcam
