# modify video frame file names so that opencv can read them
import os
import glob
import pathlib
import re
import csv

def natural_order(file):
    r=re.match('(\d).(\d+).jpg',str(os.path.basename(file)))
    return (os.path.dirname(file),int(r.group(2)))

src="/home/stas/Projects/faces/YouTubeFaces/frame_images_DB"
dest="/home/stas/Projects/faces/YouTubeFacesQQQ/frame_images_DB"

# load face boxes
boxes={}
dirs=glob.glob(src+'/*/')
for dir in dirs:
    path=dir[:-1]+'.labeled_faces.txt'
    print(path)
    with open(path, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in reader:
            fname=row[0]
            box=list(map(int,row[2:6]))
            boxes[src+'/'+fname.replace('\\','/')]=box
#print(boxes)
#quit()
# rename frame files
dirs=glob.glob(src+'/*/*/')
for dir in dirs:
    _,_,testcase=dir.partition(src)
    print('testcase',testcase)
    images=glob.glob(src+testcase+"*.jpg")
    #print(images[0])
    pathlib.Path(dest+testcase).mkdir(parents=True, exist_ok=True)
    i=0
    for src_image in sorted(images, key=natural_order):
        dest_image=dest+testcase+'%05d.jpg' % i
        dest_box=dest+testcase+'%05d.txt' % i
        box=boxes[src_image]
        #print(box)
        with open(dest_box,'w') as f:
            f.write(str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3]))
        #print(src_image, dest_image)
        os.symlink(src_image, dest_image)
        i+=1

quit()
