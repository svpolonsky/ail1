import glob
import os
from collections import Counter

data_dir="/home/stas/Projects/faces/myapp1/FaceDB"
ref_dir="/home/stas/Projects/faces/YouTubeFaces_05d/frame_images_DB"

#-------------------------------------------------------------------------------
# cluster/class ratio
#-------------------------------------------------------------------------------

data_clusters=glob.glob(data_dir+'/*/')
ref_clusters=glob.glob(ref_dir+'/*/')
print("clusters",len(data_clusters),"classes",len(ref_clusters),"cluster/class","{0:0.2f}".format(float(len(data_clusters))/float(len(ref_clusters))) )

#-------------------------------------------------------------------------------
# Purity measure, as defined in https://en.wikipedia.org/wiki/Cluster_analysis
#-------------------------------------------------------------------------------


def compute_purity(mypath):
    total=0
    major=0
    for cluster in os.listdir(mypath):
        cpath=os.path.join(mypath, cluster)
        if os.path.isdir(cpath):
            #print(cluster)
            true_names=[]
            for record in os.listdir(cpath):
                rpath=os.path.join(cpath, record)
                if os.path.isdir(rpath):
                    #print("   ",record)
                    fname=os.path.join(rpath, 'true_name.txt')
                    #print(fname)
                    with open(fname,"r") as f:
                        true_name=f.read()
                        #print(true_name)
                    true_names.append(true_name.rstrip())
                    total+=1
            #print(true_names)
            count=Counter(true_names)
            mc=count.most_common()
            name,number=mc[0]
            major+=number
    return float(major)/float(total)

purity=compute_purity(data_dir)
print("purity","{0:0.2f}".format(purity))
