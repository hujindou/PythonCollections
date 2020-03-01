import sys
import os
import dlib
import glob
import numpy as np

fileExtensions = ['*.jpeg', '*.png', '*.jpg']

if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./calcImage128D.py imagePath debug\n"
        )
    exit()

imagePath = sys.argv[1]
debugFlag = sys.argv[2] == "debug"

if not os.path.isdir(imagePath):
    print("Directory not exist\n")
    exit()
    pass

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

win = None

if debugFlag:
    win = dlib.image_window()
    pass

totalProcessImage = 0

for tmpFileExtension in fileExtensions:

    for f in glob.glob(os.path.join(imagePath, "**/", tmpFileExtension), recursive=True):

        print("Begin processing file ->", f)

        totalProcessImage += 1

        img = dlib.load_rgb_image(f)

        dets = detector(img, 1)

        print("Number of faces detected: {}".format(len(dets)))

        if len(dets) > 1:
            print("More than one face detected ... Skip")
            continue

        for k, d in enumerate(dets):
            shape = sp(img, d)

            if debugFlag:
                win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)
                pass

            face_descriptor = facerec.compute_face_descriptor(img, shape)

            nparr = np.asarray(face_descriptor)
            with open(os.path.join(imagePath, "128Data.txt"), "ab") as filewriter:
                np.savetxt(filewriter, nparr, newline=" ")
                filewriter.write(b"\n")
                pass
            pass

        if debugFlag:
            dbgimg = dlib.load_rgb_image(f)
            dbgdets, scores, idx = detector.run(dbgimg, 1, -1)
            for i, d in enumerate(dbgdets):
                print("Debug Info ## Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
                pass
            pass

        if debugFlag:
            dlib.hit_enter_to_continue()
            pass

        pass
    pass

print(totalProcessImage, "image processed")
