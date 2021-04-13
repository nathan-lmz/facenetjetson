import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
import cv2
import time
import warnings

import tensorflow as tf
from tensorflow.keras.models import load_model
# from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skimage.transform import resize
import tensorflow.compat.v1 as tf1

from scipy.spatial import distance
import json
import csv
import pickle
import pandas as pd

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# haar_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
haar_path = "./models/haarcascade_frontalface_alt2.xml"
train_data_path = './data/train/'
names = []
unknownid = 0
train_folders = os.listdir("./data/train")
for i in train_folders:
    if not os.path.isdir(i):
        names.append(i)
print(names)
for i in names:
    if (i=="0Unknown"):
        break
    unknownid+=1
print(unknownid)
image_size = 160

model_path = './pb/fn_fp16_128MB.pb'

with tf.compat.v1.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    em = tf1.import_graph_def(graph_def, input_map = None, name='',return_elements=["Bottleneck_BatchNorm/batchnorm_1/add_1:0"])
#     tf1.import_graph_def(graph_def, input_map = None, name='')
    
# Get input and output tensors
images_placeholder = tf1.get_default_graph().get_tensor_by_name("input_1:0")

embeddings = tf1.get_default_graph().get_tensor_by_name("Bottleneck_BatchNorm/batchnorm_1/add_1:0")

embedding_size = embeddings.get_shape()[1]
print(embedding_size)

def pkl_save(data,path):
    file = open(path,'wb')
    pickle.dump(data, file)
    file.close()

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(haar_path)
    
    aligned_images = []
    for filepath in filepaths:
        print(filepath)
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.05,
                                         minNeighbors=5)
        (x, y, w, h) = faces[0]
        cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
            
    return np.array(aligned_images)

def calc_embs(filepaths, name, margin=10, batch_size=1):
    gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    with tf.compat.v1.Session(config=tf1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
#         if (name!="0Unknown"):
        aligned_images = prewhiten(load_and_align_images(filepaths, margin))
#         if(name=="0Unknown"):
#             aligned_images = load_and_align_images(filepaths, margin)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
#             feed_dict = { images_placeholder:aligned_images[start:start+batch_size], phase_train_placeholder:False }
            feed_dict = { images_placeholder:aligned_images[start:start+batch_size]}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            pd.append(emb_array)
    #         pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
        embs = l2_normalize(np.concatenate(pd))
    
    return embs

def inf_embs(img, margin=10, batch_size=1):
    aligned_images = prewhiten(img)
    pd = []
    feed_dict = { images_placeholder:aligned_images}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    pd.append(emb_array)
    embs = l2_normalize(np.concatenate(pd))
    
    return embs

def calc_dist(img_name0, img_name1):
    return distance.euclidean(data[img_name0]['emb'], data[img_name1]['emb'])

def calc_infer_dist(img_name0, inf_emb):
    return distance.euclidean(data[img_name0]['emb'], inf_emb)

def train_svm(dir_basepath, names, max_num_img=50):
    labels = []
    embs = []
    data = {}
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths, name)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        for i in range(len(filepaths)):
            data['{}{}'.format(name, i)] = {'image_filepath' : filepaths[i],
                                            'emb' : embs_[i]}
        
    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    pkl_save(embs,'./pkl/embs.pkl')
    pkl_save(y,'./pkl/y.pkl')
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    return le, clf, data

def train_softmax(dir_basepath, names, max_num_img=10):
    labels = []
    embs = []
    data = {}
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths, name)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        for i in range(len(filepaths)):
            data['{}{}'.format(name, i)] = {'image_filepath' : filepaths[i],
                                            'emb' : embs_[i]}
        
    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    pkl_save(embs,'./pkl/embs.pkl')
    pkl_save(y,'./pkl/y.pkl')
    clf = LogisticRegression()
    clf.fit(embs, y)
    return le, clf, data

def infer(le, clf, img):
    embs = inf_embs(img)
    clfpred = clf.predict(embs)
    clfprob = clf.predict_proba(embs)
    pred = le.inverse_transform(clfpred)
    if clfprob[0][unknownid]>clfprob[0][clfpred[0]]:
        pred[0]="0Unknown"

    data_name = pred[0]+'1'
    dist = distance.euclidean(data[data_name]['emb'], embs[0])
    if(dist>0.88):
         pred[0]="0Unknown"
            
    print(pred[0]+"  "+str(dist))
    
    return pred

start = time.time()

data_file = open('./pkl/data.pkl','rb')
data = pickle.load(data_file)

le_file = open('./pkl/le.pkl','rb')
le = pickle.load(le_file)

y_file = open('./pkl/y.pkl','rb')
y = pickle.load(y_file)

embs_file = open('./pkl/embs.pkl','rb')
embs = pickle.load(embs_file)
clf = SVC(kernel='linear', probability=True).fit(embs, y)

print(str(time.time()-start))

print(model_path)

import cv2
import time

cam_height = 480
cam_width = 640

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

margin = 10

rec = []
temprec = []
showtext = "Last attendance taken:"
record_dir = "./record/"+time.strftime("%Y-%m-%d",time.localtime())+".csv"

# print(str(time.time()-qdtime))

gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=0.33, allow_growth=True)
# gpu_options = tf1.GPUOptions(allow_growth=True)

with tf.compat.v1.Session(config=tf1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
    
    
    while(True):
        start = time.time()

        ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,minNeighbors=12, minSize=(210,210))

        aligned_images = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped = frame[y-margin//2:y+h+margin//2, x-margin//2:x+w+margin//2, :]
            
            if cropped.size>0:
                aligned = resize(cropped, (image_size, image_size), mode='reflect')
                aligned_images.append(aligned)
                al=np.array(aligned_images)
                pred = infer(le, clf, al)
                cv2.putText(frame, pred[0], (x, y+h+55), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 1, cv2.LINE_AA)
                temprec.append(pred[0])
        if(len(temprec)>1) and (temprec[0]!=temprec[1]):
            temprec=[]
#         if(len(temprec)>2):
#             if(temprec[1]!=temprec[2]):
#                 temprec=[]
#             if(len(temprec)>3):
#                 if(temprec[0]!='0Unknown') and (temprec[2]==temprec[3]):
#     #                 if(temprec[0]==temprec[1]) and (temprec[1]==temprec[2]):
#                     rec.append([temprec[0],time.strftime("%H:%M:%S",time.localtime())])
#                 temprec=[]
        if(len(temprec)>2):
            if(temprec[0]!='0Unknown') and (temprec[1]==temprec[2]):
    #                 if(temprec[0]==temprec[1]) and (temprec[1]==temprec[2]):
                rec.append([temprec[0],time.strftime("%H:%M:%S",time.localtime())])
            temprec=[]
        
        end = time.time()
        t = end-start
        fps="FPS: "+ str(format(1/t, '.2f'))
        cv2.putText(frame,fps, (30, cam_height-25), cv2.FONT_HERSHEY_DUPLEX, 1, (80, 250, 0), 1, cv2.LINE_AA)
        cv2.putText(frame,showtext, (6, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 20, 240), 1, cv2.LINE_AA)
        reclen = len(rec)
        if(reclen>0):
            cv2.putText(frame,rec[reclen-1][0], (5, 67), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 240, 240), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open(record_dir, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rec)
            break
        
        
cap.release()
cv2.destroyAllWindows()