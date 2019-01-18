import os
import pickle

def read_encodings(mypath):
    print("read_encodings",mypath)
    encodings=[]
    names=[]
    for customer in os.listdir(mypath):
        cpath=os.path.join(mypath, customer)
        if os.path.isdir(cpath):
            #print(customer)
            for record in os.listdir(cpath):
                rpath=os.path.join(cpath, record)
                if os.path.isdir(rpath):
                    #print(record)
                    fname=os.path.join(rpath, 'encoding.txt')
                    #print(fname)
                    with open(fname,"rb") as f:
                        encoding=pickle.loads(f.read())
                        encodings.append(encoding)
                        names.append(customer)
    return names, encodings
