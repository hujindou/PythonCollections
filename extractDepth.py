#OpenCV 4.20
#Intel Realsense 2

#dlib cnn face detector is too slow
#128D data from image is the same to 128D data from the aligned version image
#128D data from 5_face is not equal 128D data from 68 face

#opencv dnn face + dlib face landmarks is not stable
#than dlib face detector + dlib face landmarks
from __future__ import absolute_import, division, print_function, unicode_literals

import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd

import dlib

import os
import time

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

CAMWIDTH = 1280
CAMHEIGHT = 720

detector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
dlibcnnfacedetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
dlibfacerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

#caffeNet = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
tfNet = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
torchNet = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

#torchNetRealFaceCheck = cv2.dnn.readNetFromTorch("/home/hjd/realsensenet.pth")
#tfRealFaceCheck = cv2.dnn.readNetFromTensorflow("/home/hjd/tfmodel/saved_model.pb")

onnxnet = cv2.dnn.readNetFromONNX("/home/hjd/alexnet.onnx")

import cProfile

pr = cProfile.Profile()
pr.enable()

def GetResizeAndPaddingData2(pNparray):
    #pDataFrame = filterDepth(pNparray)
    pDataFrame = pNparray
    expectRowCount = 64
    expectColCount = 64
    currentRowCount = pDataFrame.shape[0]
    currentColCount = pDataFrame.shape[1]
    expectRatio = expectRowCount / expectColCount
    currentRatio = currentRowCount / currentColCount
    resizeRowCount = expectRowCount
    resizeColCount = expectColCount
    if currentRatio >= expectRatio:
        resizeColCount = int(resizeRowCount / currentRatio)
    else:
        resizeRowCount = int(resizeColCount * currentRatio)
    paddingTop = (expectRowCount - resizeRowCount) // 2
    paddingBot = expectRowCount - resizeRowCount - paddingTop
    paddingLft = (expectColCount - resizeColCount) // 2
    paddingRgt = expectColCount - resizeColCount - paddingLft

    if paddingTop + paddingBot + resizeRowCount > expectRowCount:
        print(expectRowCount, expectColCount, currentRowCount, currentColCount)
        return None
    if paddingLft + paddingRgt + resizeColCount > expectColCount:
        print(expectRowCount, expectColCount, currentRowCount, currentColCount)
        return None

    rszimg = cv2.resize(pDataFrame.astype('float64'), (resizeColCount, resizeRowCount))
    resimg = cv2.copyMakeBorder(rszimg,
                                paddingTop, paddingBot, paddingLft, paddingRgt,
                                cv2.BORDER_CONSTANT,
                                0)

    if resimg.shape[0] > expectRowCount or resimg.shape[1] > expectColCount:
        print(resizeRowCount, resizeColCount, paddingTop, paddingBot, paddingLft, paddingRgt)
        print("copyMakeBorder Error", rszimg.shape[0], rszimg.shape[1],
              expectRowCount, expectColCount, currentRowCount, currentColCount)
        return None

    return filterDepth(resimg)

def filterDepth(np2darr):
    res = np.zeros((np2darr.shape[0], np2darr.shape[1]))
    binSum, binAnchor = np.histogram(np2darr.reshape(-1))
    binCount = len(binSum)
    minAnchor = binAnchor[0]
    maxAnchor = binAnchor[-1]
    zeroCount = (np2darr == 0).sum()
    for tmp in range(binCount):
        if tmp < binCount - 1:
            if (binSum[tmp] + binSum[tmp + 1]) / (binSum.sum() - zeroCount) > 0.9:
                minAnchor = binAnchor[tmp]
                maxAnchor = binAnchor[tmp + 1 + 1]

                # check one more bin
                if tmp + 1 + 1 < binCount and binSum[tmp + 1 + 1] > 0:
                    maxAnchor = binAnchor[tmp + 1 + 1 + 1]
                    pass

                pass
            pass
        pass

    # make value bigger than maxAnchor or value smaller than minAnchor = 0
    tmpTop = None
    tmpBot = None
    tmpLeft = None
    tmpRight = None
    #print("filter anchor is ", minAnchor, maxAnchor)
    for j in range(res.shape[0]):
        for i in range(res.shape[1]):
            res[j][i] = np2darr[j][i]
            if j - 1 >= 0:
                tmpTop = np2darr[j - 1][i]
            else:
                tmpTop = None

            if j + 1 < res.shape[0]:
                tmpBot = np2darr[j + 1][i]
            else:
                tmpBot = None

            if i - 1 >= 0:
                tmpLeft = np2darr[j][i - 1]
            else:
                tmpLeft = None

            if i + 1 < res.shape[1]:
                tmpRight = np2darr[j][i + 1]
            else:
                tmpRight = None

            if res[j][i] >= minAnchor and res[j][i] <= maxAnchor:
                # do nothing
                pass
            else:
                tmplist = []

                if tmpLeft is not None and tmpLeft >= minAnchor and tmpLeft <= maxAnchor:
                    tmplist.append(tmpLeft)
                if tmpTop is not None and tmpTop >= minAnchor and tmpTop <= maxAnchor:
                    tmplist.append(tmpTop)
                if tmpRight is not None and tmpRight >= minAnchor and tmpRight <= maxAnchor:
                    tmplist.append(tmpRight)
                if tmpBot is not None and tmpBot >= minAnchor and tmpBot <= maxAnchor:
                    tmplist.append(tmpBot)

                #print(tmplist)

                if len(tmplist) >= 3:
                    res[j][i] = sum(tmplist) / len(tmplist)
                else:
                    res[j][i] = 0

                pass
            pass
        pass

    ajust = res.max()
    res = ajust - res
    res[res == ajust] = 0
    #tmpajust2 = res(res != ajust).max()

    return res

import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 169 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


torchNetRealFaceCheck = torch.load("/home/hjd/realsensenet.pth")
torchNetRealFaceCheck.eval()

tfRealFaceCheck = tf.keras.models.load_model('/home/hjd/tfmodel.h5')
tfRealFaceCheck.summary()
probability_model = tf.keras.Sequential([tfRealFaceCheck,
                                         tf.keras.layers.Softmax()])


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

#not use depth data
config.enable_stream(rs.stream.depth, CAMWIDTH, CAMHEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, CAMWIDTH, CAMHEIGHT, rs.format.bgr8, 30)

#calculate display width and height
DSPWIDTH = 720
DSPHEIGHT = 720

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

#align depth info to color info for purpose of increasing accuracy
align_to = rs.stream.color
align = rs.align(align_to)

imageIndex = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #print("DepthData", depth_image.shape, depth_image.dtype)
        #print("ColorData", color_image.shape, color_image.dtype)

        #flip image
        color_image = cv2.flip(color_image, 1)
        color_image = color_image[0:720, 280:280+DSPWIDTH]
        depth_image = cv2.flip(depth_image, 1)
        depth_image = depth_image[0:720, 280:280 + DSPWIDTH]

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        #images = np.vstack((color_image, depth_colormap))

        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0), False, False)

        #print(blob.shape, blob.dtype)

        tfNet.setInput(blob)
        detections = tfNet.forward()
        #print(detections.shape, detections.dtype, detections.shape[2])

        images = color_image
        #images = cv2.resize(color_image, (300, 300))

        w = DSPWIDTH
        h = DSPHEIGHT

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                tmparr = depth_image[startY:endY, startX:endX]
                # print(tmparr.shape,tmparr.dtype)

                # predictions = probability_model.predict(
                #     GetResizeAndPaddingData2(tmparr).
                #         astype(np.float32).
                #         reshape(1,64,64,1))

                # print(np.argmax(predictions[0]))

                #outputs = torchNetRealFaceCheck(torch.from_numpy(
                #    GetResizeAndPaddingData2(tmparr).astype(np.float32).reshape(1, 1,64,64)))
                #_, predicted = torch.max(outputs, 1)
                #print(predicted)


                onnxnet.setInput(GetResizeAndPaddingData2(tmparr).astype(np.float32).reshape(1, 1, 64, 64))
                onnxRes = onnxnet.forward()
                _, predicted = torch.max(torch.tensor(onnxRes), 1)
                # print(predicted)

                rectColor = (0, 0, 255)

                if predicted[0] == 0:
                    rectColor = (0, 255, 0)

                # if np.argmax(predictions[0]) == 0:
                #     rectColor = (0, 255, 0)

                imageIndex += 1

                # with open("/home/hjd/depthData/" + str(imageIndex) + ".txt", "ab") as f:
                #     # numpy array having more than one row , use delimiter
                #     np.savetxt(f, tmparr, fmt='%i', delimiter=" ")
                #     f.write(b"\n")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(images, (startX, startY), (endX, endY),
                              rectColor, 2)
                cv2.putText(images, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                pass
            pass

        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, images)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', np.hstack((images, depth_colormap)))

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()
    pr.disable()
    # after your program ends
    pr.print_stats(sort="tottime")