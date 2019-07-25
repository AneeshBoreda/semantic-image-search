
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.activations.relu import relu
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
