#OpenCV 4.20
#Intel Realsense 2

#dlib cnn face detector is too slow
#128D data from image is the same to 128D data from the aligned version image
#128D data from 5_face is not equal 128D data from 68 face

#opencv dnn face + dlib face landmarks is not stable
#than dlib face detector + dlib face landmarks

import os
import pyrealsense2 as rs
import numpy as np
import cv2
import dlib

CAMWIDTH = 1280
CAMHEIGHT = 720

def collectDataUsingOpenCvDNN(dataSavePath = "/home/hjd/depthData/", saveBlank = False):
    tfNet = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # not use depth data
    config.enable_stream(rs.stream.depth, CAMWIDTH, CAMHEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, CAMWIDTH, CAMHEIGHT, rs.format.bgr8, 30)

    # calculate display width and height
    DSPWIDTH = 720
    DSPHEIGHT = 720

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # align depth info to color info for purpose of increasing accuracy
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

            # print("DepthData", depth_image.shape, depth_image.dtype)
            # print("ColorData", color_image.shape, color_image.dtype)

            # flip image
            color_image = cv2.flip(color_image, 1)
            color_image = color_image[0:720, 280:280 + DSPWIDTH]
            depth_image = cv2.flip(depth_image, 1)
            depth_image = depth_image[0:720, 280:280 + DSPWIDTH]

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0), False, False)

            # print(blob.shape, blob.dtype)

            tfNet.setInput(blob)
            detections = tfNet.forward()
            # print(detections.shape, detections.dtype, detections.shape[2])

            images = color_image
            # images = cv2.resize(color_image, (300, 300))

            w = DSPWIDTH
            h = DSPHEIGHT

            humanFaceDetectedFlag = False

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

                    rectColor = (0, 0, 255)

                    imageIndex += 1

                    with open(os.path.join(dataSavePath, str(imageIndex) + ".txt"), "wb") as f:
                        np.savetxt(f, tmparr, fmt='%i', delimiter=" ")
                        f.write(b"\n")

                    humanFaceDetectedFlag = True

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

            #save blank data(300x300) locates center of 720x720
            if saveBlank == True:
                #if human face inside photo or screen save faceclip depth
                #   or save mid 300x300
                if humanFaceDetectedFlag == False:
                    imageIndex += 1
                    tmparr = depth_image[210:210 + 300, 210:210 + 300]
                    with open(os.path.join(dataSavePath, str(imageIndex) + ".txt"), "wb") as f:
                        np.savetxt(f, tmparr, fmt='%i', delimiter=" ")
                        f.write(b"\n")
                        pass

                    pass
                pass

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', np.hstack((images, depth_colormap)))

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:

        pipeline.stop()

    pass

def collectDataUsingDlibHog(dataSavePath = "/home/hjd/depthDataFail/", saveBlank = False):
    detector = dlib.get_frontal_face_detector()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # not use depth data
    config.enable_stream(rs.stream.depth, CAMWIDTH, CAMHEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, CAMWIDTH, CAMHEIGHT, rs.format.bgr8, 30)

    # calculate display width and height
    DSPWIDTH = 720
    DSPHEIGHT = 720

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # align depth info to color info for purpose of increasing accuracy
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

            # print("DepthData", depth_image.shape, depth_image.dtype)
            # print("ColorData", color_image.shape, color_image.dtype)

            # flip image
            color_image = cv2.flip(color_image, 1)
            color_image = color_image[0:720, 280:280 + DSPWIDTH]
            depth_image = cv2.flip(depth_image, 1)
            depth_image = depth_image[0:720, 280:280 + DSPWIDTH]

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


            images = color_image

            w = DSPWIDTH
            h = DSPHEIGHT

            humanFaceDetectedFlag = False

            resized = cv2.resize(rgb_image, (180, 180))
            dets = detector(resized, 1)
            for k, d in enumerate(dets):
                (startX, startY, endX, endY) = (d.left()*4, d.top()*4, d.right()*4, d.bottom()*4)

                tmparr = depth_image[startY:endY, startX:endX]

                rectColor = (0, 0, 255)

                imageIndex += 1

                with open(os.path.join(dataSavePath, str(imageIndex) + ".txt"), "wb") as f:
                    np.savetxt(f, tmparr, fmt='%i', delimiter=" ")
                    f.write(b"\n")

                humanFaceDetectedFlag = True

                cv2.rectangle(images, (startX, startY), (endX, endY),
                              rectColor, 2)

                pass

            if saveBlank == True:
                #if human face inside photo or screen save faceclip depth
                #   or save mid 300x300
                if humanFaceDetectedFlag == False:
                    imageIndex += 1
                    tmparr = depth_image[210:210 + 300, 210:210 + 300]
                    with open(os.path.join(dataSavePath, str(imageIndex) + ".txt"), "wb") as f:
                        np.savetxt(f, tmparr, fmt='%i', delimiter=" ")
                        f.write(b"\n")
                        pass

                    pass
                pass

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', np.hstack((images, depth_colormap)))

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:

        pipeline.stop()

    pass

import sys

if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   python3 depthDataCollection.py 11 savePath\n"
        )
    exit()

savePath = sys.argv[2]
typeFlag = sys.argv[1]

if typeFlag == "11":
    collectDataUsingOpenCvDNN(savePath, False)
    pass
elif typeFlag == "21":
    collectDataUsingOpenCvDNN(savePath, True)
    pass
elif typeFlag == "12":
    collectDataUsingDlibHog(savePath, False)
    pass
elif typeFlag == "22":
    collectDataUsingDlibHog(savePath, True)
    pass