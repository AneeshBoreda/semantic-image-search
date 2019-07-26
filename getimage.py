import numpy as np
import text





def get_image(query, idf, vocab_dict, model, data):

    query_vector = text.se_text(query, idf, vocab_dict)
    list_of_imageids = [k for k in data]
    sims = np.zeros(len(data))
    for i, image_id in enumerate(data):
        image_embedding=model(data[image_id]).data
        image_embedding/=np.linalg.norm(image_embedding)
        sims[i] = image_embedding @ query_vector
    ind = np.argpartition(sims, -4)[-4:]
    top_4 = []
    for i in ind:
        top_4.append(list_of_imageids[i])
    del list_of_imageids
    del sims
    return top_4





