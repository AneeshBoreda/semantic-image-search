import json
import numpy as np
import pickle
from urllib.request import urlopen
import matplotlib.pyplot as plt

def get_triple():

    with open('resnet18_features.pkl', 'rb') as f:
        data = pickle.load(f)

    pickle_in = open("imageid_to_captions.pickle", "rb")
    imageid_to_captions = pickle.load(pickle_in)
    pickle_in.close()

    random_index = int(np.random.randint(0, len(list_of_imageids)))
    image_id = list_of_imageids[random_index]
    good_image = data[image_id]
    caption = imageid_to_captions[image_id][int(np.random.randint(0, 5))]

    random_index = int(np.random.randint(0, len(list_of_imageids)))
    image_id = list_of_imageids[random_index]
    bad_image = data[image_id]
    return good_image, caption, bad_image
