import text

data = None
list_of_imageids = None

def load_image_ids(data1):
    data=data1
    list_of_imageids = [k for k in data]


def get_image(query, idf, vocab_dict, model):

    query_vector = text.se_text(query, idf, vocab_dict)
    sims = np.zeros(len(data))
    for i, image_id in enumerate(data):
        image_embedding=model(data[image_id])
        image_embedding/=np.linalg.norm(image_embedding)
        sims[i] = np.dot(query_vector,image_embedding)
    ind = np.argpartition(sims, -4)[-4:]

    top_4 = []
    for i in ind:
        top_4.append(list_of_imageids[i])
    return top_4





