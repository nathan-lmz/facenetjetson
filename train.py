import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
import cv2
import time

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



model_path = './pb/fn_fp16_256MB.pb'

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
        aligned_images = prewhiten(load_and_align_images(filepaths, margin))
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            feed_dict = { images_placeholder:aligned_images[start:start+batch_size]}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            pd.append(emb_array)
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
le, clf, data = train_svm(train_data_path, names)
# le, clf = train_softmax(train_data_path, names)
end = time.time()
print(str(end-start))


start = time.time()

data_file = open('./pkl/data.pkl','wb')
pickle.dump(data, data_file)
data_file.close()

le_file = open('./pkl/le.pkl','wb')
pickle.dump(le, le_file)
le_file.close()
    
print(str(time.time()-start))