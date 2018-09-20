# Record face video for myapp1 (AIL)

# I took the video capture & write code from here:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import os
import datetime
import time
import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset-directory", type=str, default="testset")
ap.add_argument("-n", "--face-name", type=str, default="Unknown")
args = vars(ap.parse_args())

def get_output_name():
    root=os.path.dirname(os.path.realpath(__file__))
    dataset=args["dataset_directory"]
    time_format = '%Y-%m-%d %H:%M:%S'
    fname=args["face_name"]+" "+str(datetime.datetime.now().strftime(time_format))
    path=os.path.join(root,dataset,fname)+".avi"
    return path

path=get_output_name()

print("[INFO] output file: "+str(path))

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(path,fourcc, 20.0, (640,480))

time.sleep(2.0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # adjust frame orientation
        frame = cv2.flip(frame,1)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        cv2.moveWindow('frame',200,200)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
