from sklearn.cluster import DBSCAN
import numpy as np
from common_functions import read_encodings
data_dir="/home/stas/Projects/faces/myapp1/FaceDB"
names,encodings=read_encodings(data_dir)
clt = DBSCAN(metric="euclidean", n_jobs=-1)
clt.fit(encodings)
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))
