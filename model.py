
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import margin_ranking_loss
import numpy as np
import mygrad as mg

class Model():
    def __init__(num_in, num_out):
        self.dense=dense(num_in, num_out, weight_initializer=glorot_normal)
    def __call__(x):
        return self.dense(x)
    @property
    def parameters():
        return self.dense.parameters

def train(model, text_emb, good_img, bad_img, optim):
    sim_to_good=sim(text_emb,model(good_img))
    sim_to_bad=sim(text_emb,model(bad_img))
    loss=margin_ranking_loss(sim_to_good,sim_to_bad,1,0.1)
    loss.backward()
    optim.step()
    loss.null_gradients()
    return loss.item()
def sim(v1, v2):
    v2/=np.linalg.norm(v2)
    return mg.cos(v1,v2)

