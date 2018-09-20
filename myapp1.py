from pyimagesearch.centroidtracker import CentroidTracker
import os
import cv2
import sys
import time
import imutils
import numpy as np
import datetime
from collections import OrderedDict
import math
# http://face-recognition.readthedocs.io/en/latest/face_recognition.html
import face_recognition
import pathlib
import uuid
import pickle
import glob
import argparse

def get_FaceDB_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),"FaceDB")


from common_functions import read_encodings


def track_faces(frame,W,H,net,confidence,ctrack):
    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    #print("rects", len(rects))
    objects = ctrack.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return frame,objects,rects;

def centroid(rect):
    (startX,startY,endX,endY)=rect
    cX=int((startX+endX)/2.0)
    cY=int((startY+endY)/2.0)
    return [cY,cY]

time_format = '%Y-%m-%d %H:%M:%S'

def write_encoding(id,encoding,pic,true_name):
    dt=str(datetime.datetime.now().strftime(time_format))
    #dir=pathlib.Path.cwd().joinpath('FaceDB',id,dt)
    dir=os.path.join(get_FaceDB_dir(),id,dt)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    fname=pathlib.Path(dir).joinpath('encoding.txt')
    with open(fname,'wb') as f:
        f.write(pickle.dumps(encoding))
    fname2=str(pathlib.Path(dir).joinpath('face.png'))
    #print(fname2)
    cv2.imwrite(fname2,pic)
    fname3=str(pathlib.Path(dir).joinpath('true_name.txt'))
    with open(fname3,'w') as f:
        f.write(true_name)
    return 0

# list of face update times
def face_report(name):
    print("Reporting on",name)
    records=[]
    dir=os.path.join(get_FaceDB_dir(),name)
    for record in os.listdir(dir):
        records.append(datetime.datetime.strptime(str(record), time_format))
    print("Number of records",len(records))
    records.sort()
    print('Last seen on',records[-1].strftime(time_format))
    print('First seen on',records[0].strftime(time_format))
    return records

def update_face(id,encoding,pic,true_name):
    global names
    global encodings
    names.append(id)
    encodings.append(encoding)
    write_encoding(id,encoding,pic,true_name)
    return 0

def register_face(encoding,pic,true_name):
    global names
    global encodings
    id=str(uuid.uuid1())
    names.append(id)
    encodings.append(encoding)
    write_encoding(id,encoding,pic,true_name)
    return id

def recognize_faces(id,face_centroid,rects,frame,true_name):
    global names
    global encodings
    if rects==[]:
        print('face passed too quick')
        return 1
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # box: top=startY, right=endX, bottom=endY, left=startX
    #boxes = face_recognition.face_locations(rgb,model="hog")
    #print("boxes:",boxes)

    # find rect for the face
    d0=math.inf
    xf,yf=face_centroid
    #print("f:",xf,yf)
    for rect in rects:
        xr,yr=centroid(rect)
        #print("r:",xr,yr)
        d2=float(xf-xr)**2+float(yf-yr)**2
        #print(d2)
        d=math.sqrt(d2)
        if d<d0:
            d0=d
            rect0=rect

    (startX,startY,endX,endY)=rect0
    fheight,fwidth=frame.shape[:2]
    print("rect0",rect0)
    if startX>fwidth or endX>fwidth or startY>fheight or endY>fheight or startX<0 or startY<0 or endX<0 or endY<0:
        print("weird rect0 in recognize_faces")
        print(rect0,"frame",frame.shape[:2])
        return 1

    pic=frame[startY:endY,startX:endX,:]
    if False:
        cv2.imshow('Face',pic)
        cv2.moveWindow('Face',200,600)
    # box: top=startY, right=endX, bottom=endY, left=startX
    box=(startY,endX,endY,startX)
    encoding = face_recognition.face_encodings(rgb, [box])[0]
    # lower tolerance is more strict. Default 0.6
    # this number should be optimized to get good cluster/class ratio
    tolerance=0.51
    matches=face_recognition.compare_faces(encodings,encoding,tolerance)
    if True in matches:
        matchedIdxs=[i for (i,b) in enumerate(matches) if b]
        counts={}
        print("matches",matchedIdxs)
        for i in matchedIdxs:
            name=names[i]
            counts[name]=counts.get(name,0)+1
        name=max(counts,key=counts.get)
    else:
        name="Unknown"
    #print("Face",id,"recognized as",name)
    # register or update the face
    if name=="Unknown":
        new_name=register_face(encoding,pic,true_name)
        print("Registered new face",new_name)
    else:
        update_face(name,encoding,pic,true_name)
        print("Updated records for known face",name)
        face_report(name)
    return 0



# load our serialized model from disk
prototxt="deploy.prototxt"
model="res10_300x300_ssd_iter_140000.caffemodel"
#print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(prototxt, model)


# builtin camera capture 0
#video_input=0


def process_video(video_input):
    true_name=video_input.split("/")[-3]
    print(true_name,video_input)
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    (H, W) = (None, None)
    video_capture = cv2.VideoCapture(video_input)
    time.sleep(2.0)
    time1=datetime.datetime.now()

    #print("[INFO] loading encodings...")
    #names,encodings=read_encodings(get_FaceDB_dir())

    ct = CentroidTracker()

    # new faces
    faceIdCounter=OrderedDict()
    newFaceID=set()
    frameCounter = 0

    while (video_capture.isOpened()):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret==False:
            #print("ret")
            break
        frameCounter += 1
        #print("frame",frameCounter)
        frame = imutils.resize(frame, width=400)
        #rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #boxes = face_recognition.face_locations(rgb,model="hog")
        #print("boxes:",boxes)
        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        frame_copy=frame.copy()
        confidence=0.5
        frame,objects,rects=track_faces(frame,W,H,net,confidence,ct)

        # trigger RecognizeFaces based on face persistance
        newIds=set(objects.keys()).difference( set(faceIdCounter.keys()))
        for id in newIds:
            faceIdCounter[id] = 1

        oldIds=set(objects.keys()).intersection( set(faceIdCounter.keys()))
        for id in oldIds:
            faceIdCounter[id] += 1

        # I have to track the face for trigger_count frames before taking an action
        trigger_count=10
        for id, count in faceIdCounter.items():
            if count == trigger_count:
                #print(id," ",objects[id])
                print('Triggering action on persistant face')
                recognize_faces(id,objects[id],rects,frame_copy,true_name)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        cv2.moveWindow('Video',200,200)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    time2=datetime.datetime.now()
    dtime=time2-time1
    print("fps",frameCounter/(dtime.seconds+0.001))
        # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def resize_image(W,img,box):
    # start, end - face box
    height, width = img.shape[:2]
    scale = W/width
    H = int(height*scale)
    newimg = cv2.resize(img,(W,H))
    newbox = scale*box
    return newimg, newbox.astype("int")

def process_image(video_input):
    # person's name
    true_name=video_input.split("/")[-3]
    print(true_name,video_input)
    (H, W) = (None, None)
    video_capture = cv2.VideoCapture(video_input+".jpg")

    #print("[INFO] loading encodings...")
    #names,encodings=read_encodings(get_FaceDB_dir())


    # new faces
    frameCounter=0
    while (video_capture.isOpened()):
        box_file=(video_input+".txt") % frameCounter
        print("aa", box_file)
        with open(box_file,'r') as f:
            line=f.readline()
            print(line)
        cx,cy,dx,dy=map(float,line.split(','))
        box=np.array([cx-dx/2, cy-dy/2, cx+dx/2, cy+dy/2]).astype("int")
        # Capture just the first frame
        ret, frame = video_capture.read()
        if ret==False:
            break
        frame,box=resize_image(600,frame,box)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        frame_copy=frame.copy()
        #frame,objects,rects=track_faces(frame,W,H,net,confidence,ct)

        #print('Triggering action on persistant face')
        recognize_faces(0,centroid(box),[box],frame_copy,true_name)

        # Display the resulting frame
        x1,y1,x2,y2=box
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0, 255, 0), 2)
        if False:
            cv2.imshow('Video', frame)
            cv2.moveWindow('Video',200,200)
            cv2.waitKey(1000)
        break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    video_capture.release()
    cv2.destroyAllWindows()


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--list", type=str, help="file with stream list")
ap.add_argument("-s", "--stream", type=str, help="stream")
args = vars(ap.parse_args())

if args["stream"] is not None:
    video_inputs = [args["stream"]]
elif args["list"] is not None:
    fname=args["list"]
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() + '/%05d' for x in content]
    video_inputs=content
else:
    video_inputs=[]

print("[INFO] loading encodings...")
names,encodings=read_encodings(get_FaceDB_dir())

for video_input in video_inputs:
    #process_video(video_input)
    process_image(video_input)
