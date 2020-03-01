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

        #print("DepthData", depth_image.shape, depth_image.dtype)
        #print("ColorData", color_image.shape, color_image.dtype)

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

                tmparr = depth_image[startY:endY, startX:endX]
                print(tmparr)

                with open("/home/hjd/opencvFace2.txt", "ab") as f:
                    np.savetxt(f, tmparr, newline=" ")
                    f.write(b"\n")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(images, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(images, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

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