# use build-in camera
# python main.py -c 0
# use image dataset
# option -l specifies file that stores image paths
# python main.py -l small_test.txt

from pyimagesearch.centroidtracker import CentroidTracker
import os
import cv2
import sys
import time
import imutils
import copy
import numpy as np
import datetime
from collections import OrderedDict
from collections import deque
import math
# http://face-recognition.readthedocs.io/en/latest/face_recognition.html
import face_recognition
import pathlib
import uuid
import pickle
import glob
import argparse
import tkinter as tk
import PIL.Image, PIL.ImageTk
import sqlite3
from crmdb import *


def get_FaceDB_dir():
    #return os.path.join(os.path.dirname(os.path.realpath(__file__)),"FaceDB")
    return os.path.expanduser(args["database"])


from common_functions import read_encodings

def track_faces_precomputed(box_file):
    # I take bounding boxes from file instead of computing them
    box=read_bounding_box(box_file)
    faces = OrderedDict()
    faces['id0']=box
    return faces


def track_faces_nographics(frame,confidence):
    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    global facenet
    global ctrack
    H, W = frame.shape[:2]
    #print("frame shape", height, width)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0,177.0,123.0))

    facenet.setInput(blob)
    detections = facenet.forward()

    rects = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            # use the box only if it is within frame
            (startX, startY, endX, endY) = box.astype("int")
            if 0<=startX<W and 0<=endX<W and 0<=startY<H and 0<=endY<H:
                rects.append(box.astype("int"))


    # update our centroid tracker using the computed set of bounding box rectangles
    objects = ctrack.update(rects)
    # loop over the tracked objects
    # print("track_faces_nographics: number of objects",len(objects))
    faces = OrderedDict()
    for id,centroid in objects.items():
        rect=match_centroid2rects(centroid,rects,frame)
        faces[id]=rect
    return faces


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
    # all files are written in this directory, return this as a function value
    dir=os.path.join(get_FaceDB_dir(),id,dt)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    # write face encoding
    fname=pathlib.Path(dir).joinpath('encoding.txt')
    with open(fname,'wb') as f:
        f.write(pickle.dumps(encoding))
    # write face picture
    fname2=str(pathlib.Path(dir).joinpath('face.png'))
    # make sure the pic is BGR
    bgr = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fname2,bgr)
    # write true name (if known)
    fname3=str(pathlib.Path(dir).joinpath('true_name.txt'))
    with open(fname3,'w') as f:
        f.write(true_name)
    # return directory
    return dir

def gui_annotate_video(self,frame,faces):
    # scale frame and face boxes
    sframe,sboxes=resize_image_H(video_height,frame,faces.values())
    self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(sframe))
    self.canvas.delete("all")
    self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
    self.canvas.faces=[]
    for b,id in zip(sboxes,faces.keys()):
        self.canvas.faces.append(dict(id=id, box=b))
        # select the color for face's box
        if id in self.face_id:
            # recognized face
            hint=recognized_face_hint[self.face_id.index(id)]
            if hint is "DEAL":
                color=color_face_deal
            elif hint is "DEALED":
                color=color_face_dealed
            else:
                color=color_face_recognized
        else:
            # un-recognized face
            color=color_face_unrecognized
            hint=id #None
        x1,y1,x2,y2=b
        self.canvas.create_rectangle(x1,y1,x2,y2,outline=color,width=4)
        # I need to compute the hint
        if hint is not None:
            self.canvas.create_text(x1,y2,text=hint,font="Verdana 30",fill=color,anchor="nw")
    return 0


def button_deal_callback(customer_id,win):
    record_transaction(customer_id,"DEAL")
    # update hint for this customer
    idx=recognized_face_name.index(customer_id)
    recognized_face_hint[idx]="DEALED"
    win.destroy()
    return 0

def gui_suggest_deal_dialog(name):
    if name is None:
        return
    # check if the name can get a deal
    if name in recognized_face_name:
        idx=recognized_face_name.index(name)
        hint=recognized_face_hint[idx]
        if hint != "DEAL":
            print("(1) Not enough SHOW for DEAL or (2) already got a DEAL")
            return
    win=tk.Toplevel()
    win.geometry("1200x800+200+200")
    win.title("Suggest-a-deal dialog")
    win.rowconfigure(0,weight=1)
    win.rowconfigure(1,weight=1)
    win.columnconfigure(0,weight=1)
    win.columnconfigure(1,weight=1)
    cface=tk.Button(win, text="cface")
    cface.grid(row=0,column=0,sticky="EWSN")
    face2button(name,cface,0)
    pface=tk.Button(win, text="pface")
    pface.grid(row=0,column=1,sticky="EWSN")
    face2button(name,pface,1)
    # record "DEAL" transaction as a callback
    deal=tk.Button(win, text="deal",font=myfont,command = lambda name=name, win=win: button_deal_callback(name,win))
    deal.grid(row=1,column=0,sticky="EWSN")
    # remove the dialog
    cancel=tk.Button(win, text="cancel",font=myfont,command=win.destroy)
    cancel.grid(row=1,column=1,sticky="EWSN")
    return

def dialog_currentface(n):
    global recognized_face_name
    # n is the number of button
    gui_suggest_deal_dialog(recognized_face_name[n])

#def dialog_previousface(n):
    # n is the number of button
#    print("previous face",n)

def load_photo(path, width, height):
    original=PIL.Image.open(path)
    resized=original.resize((width, height),PIL.Image.ANTIALIAS)
    photo=PIL.ImageTk.PhotoImage(resized)
    return photo

def face2button(name,button,i):
    if name is None: return 0
    dir=os.path.join(get_FaceDB_dir(),name)
    # sort subdirectories based on time of recording
    records=os.listdir(dir)
    records.sort(key=lambda record: datetime.datetime.strptime(str(record), time_format), reverse=True)
    # take i-th photo (0 is the most recent)
    path=str(pathlib.Path(dir).joinpath(records[i],'face.png'))
    button.image=load_photo(path, 100, 100) # keep the reference to a photo to prevent garbage collection
    button.config(image=button.image,width=100,height=100)

def gui_face_report(name,button0):
    if name is None: return 0
    print("gui_face_report: reporting on",name)
    face2button(name,button0,0) # current face
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
    dir=write_encoding(id,encoding,pic,true_name)
    return dir

def register_face(encoding,pic,true_name):
    global names
    global encodings
    id=str(uuid.uuid1())
    names.append(id)
    encodings.append(encoding)
    write_encoding(id,encoding,pic,true_name)
    return id


def match_centroid2rects(face_centroid,rects,frame):
    xf,yf=face_centroid
    # to which rect the centroid corresponds?
    if rects==[]:
        print('no faces detected, using centroid from tracker')
        # small box is better than none ;-)
        return np.array((xf,yf,xf+1,yf+1))
    d0=math.inf
    for rect in rects:
        xr,yr=centroid(rect)
        d2=float(xf-xr)**2+float(yf-yr)**2
        d=math.sqrt(d2)
        if d<d0:
            d0=d
            rect0=rect
    (startX,startY,endX,endY)=rect0
    fheight,fwidth=frame.shape[:2]
    if startX>fwidth or endX>fwidth or startY>fheight or endY>fheight or startX<0 or startY<0 or endX<0 or endY<0:
        print("weird rect0 in match_centroid2rects")
        print(rect0,"frame",frame.shape[:2])
        return None
    return rect0

def recognize_faces(id,face_centroid,rects,frame,true_name):
    print("[INFO] recognize_faces")
    global names
    global encodings
    if rects==[]:
        print('face passed too quick')
        return 1
    #rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb=frame
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

def gui_recognize_face(id,bounding_box,frame,true_name):
    global names
    global encodings
    rgb=frame
    (startX,startY,endX,endY)=bounding_box
    fheight,fwidth=frame.shape[:2]
    if startX>fwidth or endX>fwidth or startY>fheight or endY>fheight or startX<0 or startY<0 or endX<0 or endY<0:
        print("weird bounding_box in gui_recognize_face")
        print(bounding_box,"frame",frame.shape[:2])
        return 1

    pic=frame[startY:endY,startX:endX,:]
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
        return None,None
    else:
        dir=update_face(name,encoding,pic,true_name)
        print("Updated records for known face",name)
        face_report(name)
        return dir,name


# load our serialized model from disk
prototxt="deploy.prototxt"
model="res10_300x300_ssd_iter_140000.caffemodel"
#print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(prototxt, model)

def read_bounding_box(box_file):
    # read face bounding box from a file
    with open(box_file,'r') as f:
        line=f.readline()
    cx,cy,dx,dy=map(float,line.split(','))
    box=np.array([cx-dx/2, cy-dy/2, cx+dx/2, cy+dy/2]).astype("int")
    return box

# builtin camera capture 0
#video_input=0




# number of SHOWs before the DEAL
SHOW4DEAL=2
# register new show after
MIN_SHOW_INTERVAL_SEC=10.0

def suggest_deal(customer_id):
    N=SHOW4DEAL # suggest a deal every second time
    conn=connect_CRMDB()
    # If N latest transactions come w/o deal than suggest a deal
    cursor=conn.cursor()
    sql="""
    SELECT *
        FROM (
            SELECT * FROM transactions WHERE customer_id=? ORDER BY id DESC LIMIT ?)
        WHERE transaction_type='SHOW'"""
    cursor.execute(sql,(customer_id,N,))
    results = cursor.fetchall()
    conn.close()
    return len(results)==N

def suggest_hint(customer_id):
    # how many more SHOWs before the DEAL?
    conn=connect_CRMDB()
    conn.row_factory=sqlite3.Row # format output as dictionary
    cursor=conn.cursor()
    sql="SELECT * FROM transactions WHERE customer_id=? ORDER BY id DESC LIMIT ?"
    cursor.execute(sql,(customer_id,SHOW4DEAL+1,))
    results = cursor.fetchall()
    conn.close()
    i=0
    for r in results:
        print(r['id'],r['transaction_type'])
        if r['transaction_type']=='SHOW':
            i=i+1
        else:
            break
    if i>SHOW4DEAL:
        #str="{}+/{}".format(SHOW4DEAL,SHOW4DEAL)
        str="DEAL"
    else:
        str="{}/{}".format(i,SHOW4DEAL)
    print('hint',str)
    return str



# Constants, including GUI

video_width=640
video_height=480
# I have to track the face for trigger_count frames before taking an action
trigger_count=40
face_deque_length=4
# colors would work for x-windows (rgb.txt file defines the names)
color_face_unrecognized="snow"
color_face_recognized="green"
color_face_deal="red"
color_face_dealed="yellow"
myfont=('Helvetica', '40')

# global data structures

# recognized faces
recognized_face_name=deque(maxlen=face_deque_length)
recognized_face_hint=deque(maxlen=face_deque_length) # suggestion for seller

#def video_click_callback(event):
#    print("clicked at", event.x, event.y)

class App:
    def video_callback(self,event):
        # callback when clicking on a face box
        print("video_callback: number of faces=",len(self.canvas.faces))
        for f in self.canvas.faces:
            x1,y1,x2,y2=f['box']
            if x1<=event.x<=x2 and y1<=event.y<=y2:
                idx=self.face_id.index(f['id'])
                name=recognized_face_name[idx]
                print("video_callback: found",f['id'],name)
                gui_suggest_deal_dialog(name)

    def __init__(self, window, window_title, video_source):
        global recognized_face_hint
        global recognized_face_name
        self.window = window
        self.window.geometry("1200x800+100+100")
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = video_width, height = video_height)
        self.canvas.bind("<Button-1>", self.video_callback)
        self.canvas.pack()

        # recognized faces
        self.faceIdCounter=OrderedDict()
        self.faces_frame=tk.Frame(root, borderwidth=2, relief="solid")
        self.faces_frame.pack(side="bottom", expand=True, fill="both")
        self.face_photo=deque(maxlen=face_deque_length)
        #self.face_name=deque(maxlen=face_deque_length)
        self.face_id=deque(maxlen=face_deque_length) # as given by face tracker
        self.face_button1=[]
        self.face_frame=[]
        self.faces_frame.rowconfigure(0, weight=1)
        for i in range(face_deque_length):
            self.faces_frame.columnconfigure(i, weight=1)
        for i in range(face_deque_length):
            # array of frames for recognized faces
            self.face_frame.append(tk.Frame(self.faces_frame, borderwidth=2, relief="solid"))
            self.face_frame[i].rowconfigure(0, weight=1)
            self.face_frame[i].columnconfigure(0, weight=1)
            self.face_frame[i].grid(row=0, column=i,sticky="EWSN")
            self.face_frame[i].grid_propagate(0) # prevent geometry propagation, keep specified size
            self.face_button1.append(tk.Button(self.face_frame[i], text=" ",command = lambda n=i: dialog_currentface(n)))
            self.face_button1[i].grid(row=0,column=0,sticky="EWSN")

            recognized_face_name.appendleft(None)
            recognized_face_hint.appendleft(None)
            self.face_id.appendleft(None)
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def update(self):
        global recognized_face_hint
        # Get a frame from the video source
        frameCounter = int(self.vid.get_frame_number())
        ret, frame = self.vid.get_frame()
        if ret:
            # face recogniiton should be here
            if False:
                faces=track_faces_precomputed(str(pathlib.PurePath(self.video_source).with_suffix(".txt")) % frameCounter)
            if True:
                confidence=0.5
                #(H, W) = frame.shape[:2]
                frame_copy=copy.copy(frame)
                frame1=cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
                faces=track_faces_nographics(frame1,confidence)
                #print(faces.keys())

            gui_annotate_video(self,frame,faces) # put boxes around faces
            newIds=set(faces.keys()).difference( set(self.faceIdCounter.keys()))
            for id in newIds:
                self.faceIdCounter[id] = 1
            oldIds=set(faces.keys()).intersection( set(self.faceIdCounter.keys()))
            for id in oldIds:
                self.faceIdCounter[id] += 1

            for id, count in self.faceIdCounter.items():
                if count == trigger_count:
                    print('Triggering action on persistant face',frameCounter)
                    # check if true_name exists
                    try:
                        true_name=self.video_source.split("/")[-3]
                    except AttributeError:
                        true_name='Unknown'
                    print("true_name",true_name)
                    print("tracker id",id,faces[id])
                    dir,name=gui_recognize_face(id,faces[id],frame,true_name)
                    print("dir",dir)
                    if dir is not None:
                        if name in recognized_face_name:
                            # face trackers assigned a new id to recently recognized face
                            self.face_id[recognized_face_name.index(name)]=id
                        else:
                            # Suggest Action
                            t=last_transaction_time(name)
                            t0=datetime.datetime.now()
                            duration_sec=(t0-t).total_seconds()
                            recognized_face_name.appendleft(name)
                            self.face_id.appendleft(id)
                            # CRM: interpret face recognition as SHOW transaction
                            if duration_sec>MIN_SHOW_INTERVAL_SEC:
                                print("record new SHOW:",duration_sec)
                                record_transaction(name,"SHOW")
                            hint=suggest_hint(name)
                            recognized_face_hint.appendleft(hint)
                        # update display for recognised faces
                        for i in range(face_deque_length):
                            gui_face_report(recognized_face_name[i],self.face_button1[i])
            # display frame
            self.window.after(self.delay, self.update)
        else:
            # no more frames, quit GUI
            tkquit()

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.video_source=video_source
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

    def get_frame_number(self):
        if isinstance(self.video_source,int):
            # frame number is not defined for live camera input
            n=-1
        else:
            n=self.vid.get(cv2.CAP_PROP_POS_FRAMES)
        return n

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def tkquit():
    global root
    root.destroy()

def process_video(video_input):
    true_name=video_input.split("/")[-3]
    print(true_name,video_input)
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    (H, W) = (None, None)
    video_capture = cv2.VideoCapture(video_input)
    time.sleep(2.0)
    time1=datetime.datetime.now()

    ct = CentroidTracker()

    # new faces
    faceIdCounter=OrderedDict()
    #newFaceID=set()
    frameCounter = 0

    while (video_capture.isOpened()):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret==False:
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

def resize_image_H(H,img,boxes):
    # start, end - face box
    height, width = img.shape[:2]
    scale = H/height
    W = int(width*scale)
    newimg = cv2.resize(img,(W,H))
    newboxes=[]
    #print("len boxes",len(boxes))
    for box in boxes:
        try:
            newbox=scale*np.array(box)
        except TypeError:
            print("resize_image_H: TypeError when scaling box",box,scale)
        else:
            newboxes.append(newbox.astype("int"))
    return newimg, newboxes


def resize_image(W,img,box):
    # start, end - face box
    height, width = img.shape[:2]
    scale = W/width
    H = int(height*scale)
    newimg = cv2.resize(img,(W,H))
    newbox = scale*box
    return newimg, newbox.astype("int")

def process_image(video_input):
    # I need this function to test my face classification system since:
    #  (1) video is too slow
    #  (2) vide testsets are rare, there are more image testsets
    print("[INFO] process_image")
    # I assume a string like below for video_input argument
    # /home/stas/Projects/faces/YouTubeFaces_05d/frame_images_DB/Abel_Pacheco/1/%05d.jpg

    # extract face label from the path to image
    frame_number=1
    path = pathlib.Path(video_input % frame_number)
    print("[INFO] image:",path)
    true_name=video_input.split("/")[-3]
    print("[INFO] true name:",true_name)
    (H, W) = (None, None)

    frame=cv2.imread(str(path))
    cv2.imshow('Image1', frame)
    box=read_bounding_box(str(path.with_suffix(".txt")))
    frame,box=resize_image(600,frame,box)
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    frame_copy=frame.copy()
    recognize_faces(0,centroid(box),[box],frame_copy,true_name)
    # Display the resulting frame
    x1,y1,x2,y2=box

    if True:
        cv2.imshow('Image', frame)
        #cv2.rectangle(frame, (x1,y1), (x2,y2),(0, 255, 0), 2)
        cv2.moveWindow('Image',200,200)
        time.sleep(1)
        cv2.destroyAllWindows()




ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", type=int, help="camera id")
ap.add_argument("-l", "--list", type=str, help="file with stream list")
ap.add_argument("-s", "--stream", type=str, help="stream")
ap.add_argument("-d", "--database", type=str, default="~/FaceDB",help="face database")
args = vars(ap.parse_args())

if args["camera"] is not None:
    video_inputs=[args["camera"]]
elif args["stream"] is not None:
    video_inputs = [args["stream"]]
elif args["list"] is not None:
    fname=args["list"]
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    video_inputs=content
else:
    video_inputs=[]

print("[INFO] loading encodings...")
names,encodings=read_encodings(get_FaceDB_dir())

program_mode="ImageStreamAccuracy"
#program_mode="GUI"

if program_mode=="ImageStreamAccuracy":
    print("Images")
    for video_input in video_inputs:
        process_image(video_input)
elif program_mode=="GUI":
    for video_input in video_inputs:
        print(video_input)
        print('load net')
        facenet = cv2.dnn.readNetFromCaffe(prototxt, model)
        ctrack=CentroidTracker()
        # call GUI
        root=tk.Tk()
        App(root, "AI Loyalty",video_input)
