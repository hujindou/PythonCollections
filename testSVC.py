#OpenCV 4.20
#Intel Realsense 2

#dlib cnn face detector is too slow
#128D data from image is the same to 128D data from the aligned version image
#128D data from 5_face is not equal 128D data from 68 face

#opencv dnn face + dlib face landmarks is not stable
#than dlib face detector + dlib face landmarks

import pyrealsense2 as rs
import numpy as np
import cv2

import dlib

from openFaceAlign import AlignDlib

from sklearn.svm import SVC
import pandas as pd

hjd = np.loadtxt("/home/hjd/hjd.txt")
liudehua = np.loadtxt("/home/hjd/liudehua.txt")
wangyuan = np.loadtxt("/home/hjd/wangyuan.txt")
zhangxueyou = np.loadtxt("/home/hjd/zhangxueyou.txt")
zhouhuajian = np.loadtxt("/home/hjd/zhouhuajian.txt")

dfhjd = pd.DataFrame(hjd)
dfhjd["Label"] = "hjd"
dfliudehua = pd.DataFrame(liudehua)
dfliudehua["Label"] = "Unknown"
dfwangyuan = pd.DataFrame(wangyuan)
dfwangyuan["Label"] = "Unknown"
dfzhangxueyou = pd.DataFrame(zhangxueyou)
dfzhangxueyou["Label"] = "Unknown"
dfzhouhuajian = pd.DataFrame(zhouhuajian)
dfzhouhuajian["Label"] = "Unknown"

totalpd = pd.concat([dfhjd, dfliudehua, dfwangyuan, dfzhangxueyou, dfzhouhuajian], ignore_index=True)

recognizer = SVC(C=1E6, kernel="rbf", probability=True)
recognizer.fit(totalpd.iloc[:, 0:-1], totalpd.iloc[: ,-1])

CAMWIDTH = 1280
CAMHEIGHT = 720

detector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
dlibcnnfacedetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
dlibfacerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

caffeNet = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
tfNet = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
torchNet = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

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
pipeline.start(config)

win = dlib.image_window()

align = AlignDlib()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #flip image
        color_image = cv2.flip(color_image, 1)

        color_image = color_image[0:720, 280:280+DSPWIDTH]

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        #print(crop_img.shape, crop_img.dtype)

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

        win.clear_overlay()
        win.set_image(rgb_image)

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.9:
                #print(detections[0, 0, i])
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # tmpshape = predictor(rgb_image, dlib.rectangle(startX, startY, endX, endY))
                # # print(shape)
                # # win.add_overlay(shape)
                # tmpface_descriptor = dlibfacerec.compute_face_descriptor(rgb_image, tmpshape)
                # tmparr = np.asarray(tmpface_descriptor)
                # with open("/home/hjd/opencvFace2.txt", "ab") as f:
                #     np.savetxt(f, tmparr, newline=" ")
                #     f.write(b"\n")

                # aligedFace = align.align(96,rgb_image,align.findLandmarks(tmpshape))
                # #calculate 128d with openface
                # #color_image[startY:endY, startX:endX]
                # faceBlob = cv2.dnn.blobFromImage(aligedFace, 1.0 / 255,
                #                                  (96, 96),
                #                                  (0, 0, 0), swapRB=False, crop=False)
                # torchNet.setInput(faceBlob)
                # vec = torchNet.forward()
                # vec = vec.flatten()
                # #print(vec.shape, vec.dtype)
                # with open("/home/hjd/openfacealign.txt", "ab") as f:
                #     np.savetxt(f, vec, newline=" ")
                #     f.write(b"\n")

                # face_chip = dlib.get_face_chip(rgb_image, tmpshape)
                # faceBlob = cv2.dnn.blobFromImage(face_chip, 1.0 / 255,
                #                                  (96, 96),
                #                                  (0, 0, 0), swapRB=False, crop=False)
                # torchNet.setInput(faceBlob)
                # vec = torchNet.forward()
                # vec = vec.flatten()
                # # print(vec.shape, vec.dtype)
                # with open("/home/hjd/dlibalign.txt", "ab") as f:
                #     np.savetxt(f, vec, newline=" ")
                #     f.write(b"\n")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(images, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(images, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        resized = cv2.resize(rgb_image, (180, 180))
        dets = detector(resized, 1)
        #dets = dlibcnnfacedetector(resized)
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))

            # Get the landmarks/parts for the face in box d.
            shape = predictor(rgb_image, dlib.rectangle(
                d.left()*4, d.top()*4, d.right()*4, d.bottom()*4))

            # shape2 = predictor68(rgb_image, dlib.rectangle(
            #     d.left() * 4, d.top() * 4, d.right() * 4, d.bottom() * 4))

            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                           shape.part(1)))

            # Draw the face landmarks on the screen.
            win.add_overlay(shape)
            win.add_overlay(dlib.rectangle(d.left()*4, d.top()*4, d.right()*4, d.bottom()*4))

            face_descriptor = dlibfacerec.compute_face_descriptor(rgb_image, shape)
            tmparr1 = np.asarray(face_descriptor)

            print(recognizer.predict(tmparr1.reshape(1, 128)))

            # with open("/home/hjd/dlibFace.txt", "ab") as f:
            #     np.savetxt(f, tmparr1, newline=" ")
            #     f.write(b"\n")

            # face_descriptor2 = dlibfacerec.compute_face_descriptor(color_image, shape2)
            # tmparr2 = np.asarray(face_descriptor2)
            #
            # print(np.linalg.norm(tmparr1 - tmparr2))

            #aligned face descriptor equals top
            # face_chip = dlib.get_face_chip(rgb_image, shape)
            # face_descriptor2 = dlibfacerec.compute_face_descriptor(face_chip)
            # tmparr2 = np.asarray(face_descriptor2)
            # print(np.linalg.norm(tmparr1 - tmparr2)) == 0

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()