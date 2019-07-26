import json
import numpy as np
import pickle
from urllib.request import urlopen
import matplotlib.pyplot as plt
import os.path
from os import path

#path_to_captions = "/Users/cooperbosch/Desktop/CogWorks/semantic-image-search/captions_train2014.json"
path_to_captions = "captions_train2014.json"

with open(path_to_captions, "rb") as f:
    captions_train = json.load(f)
imageid_to_captions=None
imageid_to_url=None
if not path.exists("imageid_to_captions.pickle"):
    imageid_to_captions = dict()
    for image in captions_train['images']:
        imageid_to_captions.update({image['id']: []})
    for annotation in captions_train['annotations']:
        imageid_to_captions[annotation['image_id']].append(annotation['caption'])

    pickle_out = open("imageid_to_captions.pickle", "wb")
    pickle.dump(imageid_to_captions, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("imageid_to_captions.pickle", "rb")
    imageid_to_captions = pickle.load(pickle_in)
    pickle_in.close()

if not path.exists('imageid_to_url.pickle'):
    imageid_to_url = dict()
    for image in captions_train['images']:
        imageid_to_url[image['id']] = image['coco_url']

    pickle_out = open("imageid_to_url.pickle", "wb")
    pickle.dump(imageid_to_url, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("imageid_to_url.pickle", "rb")
    imageid_to_url = pickle.load(pickle_in)
    pickle_in.close()


with open('resnet18_features.pkl', 'rb') as f:
    data = pickle.load(f)


list_of_imageids = [k for k in data]

cutoff = .8*len(list_of_imageids)
train_id = list_of_imageids[:cutoff]
test_id = list_of_imageids[cutoff:]



def get_triple(id):

    random_index = int(np.random.randint(0, len(id)))
    image_id = id[random_index]
    good_image = data[image_id]
    caption = imageid_to_captions[image_id][int(np.random.randint(0, 5))]

    random_index = int(np.random.randint(0, len(id)))
    image_id = id[random_index]
    bad_image = data[image_id]
    return good_image, caption, bad_image

def get_all_captions():

    all_captions = []
    for image_id in list_of_imageids:
        all_captions.append(' '.join(caption for caption in imageid_to_captions[image_id]))

    return all_captions

def better_get_triple():
    global index
    random_index = int(np.random.randint(0, len(list_of_imageids)))
    image_id = list_of_imageids[random_index]
    good_image = data[image_id]
    good_caption = imageid_to_captions[image_id][int(np.random.randint(0, 5))]
    idf, vocab_dict = text.save_idf(get_all_captions())

    good_image_vector = text.se_text(caption, idf, vocab_dict)

    random_indices = int(np.random.randint(0, len(list_of_imageids),10))

    bad_captions = []

    for index in random_indices:
        bad_captions.append((index, imageid_to_captions[list_of_imageids[index]][int(np.random.randint(0, 5))]))

    best = bad_caption[0[0]]
    for index, caption in bad_captions:
        if text.se_text(caption, idf, vocab_dict) @ good_image_vector > text.se_text(imageid_to_captions[best], idf, vocab_dict) @ good_image_vector):
            best = index

    image_id = list_of_imageids[best]
    bad_image = data[image_id]
    return good_image, caption, bad_image
