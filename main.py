from os import path
if not path.exists("glove.6B.50d.txt.w2v"):
    print("Error: Download the 50 dimension glove word embeddings")
    exit(0)
if not path.exists("vocab.dict"):
    print("Error: Download vocab.dict from https://github.com/AneeshBoreda/semantic-image-search")
    exit(0)
if not path.exists("idf.npy"):
    print("Error: Download idf.npy from https://github.com/AneeshBoreda/semantic-image-search")
    exit(0)
if not path.exists("model.obj"):
    print("Error: Download model.obj from https://github.com/AneeshBoreda/semantic-image -search")
    exit(0)
if not path.exists("resnet18_features.pkl"):
    print("Error: Download resnet18 features")
    exit(0)
if not path.exists("imageid_to_captions.pickle"):
    print("Error: Download imageid_to_captions.pickle from https://github.com/AneeshBoreda/semantic-image -search")
    exit(0)
if not path.exists("imageid_to_url.pickle"):
    print("Error: Download imageid_to_captions.pickle from https://github.com/AneeshBoreda/semantic-image -search")
    exit(0)

from model import *
import triplefinder as tf
import pickle
import getimage
import text
import matplotlib.pyplot as plt
import urllib.request as req

model1=load()

with open('vocab.dict','rb') as f:
    vocab_dict=pickle.load(f)
    f.close()
idf=np.load('idf.npy')
imageids=[k for k in tf.data]
inp = input("Enter text to search for, or press q to quit\n")
while inp!="q":
    ids = getimage.get_image(inp, idf, vocab_dict, model1, tf.data, imageids)
    urls = list(tf.imageid_to_url[i] for i in ids)
    fig, ax = plt.subplots(2,2)
    for i in range(4):
        f = req.urlopen(urls[i])
        img=plt.imread(f,format='jpg')
        ax[i//2,i%2].imshow(img)
    plt.show()
    inp=input("Enter text to search for, or press q to quit\n")
