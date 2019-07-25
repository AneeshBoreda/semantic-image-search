from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal


class Autoencoder:
    def __init__(self, D_in, D_out):

        self.dense1 = dense(D_in, D_out, weight_initializer=glorot_normal, bias=True)



    def __call__(self, x):

        return self.dense1(x)

    @property
    def parameters(self):

        return self.dense1.parameters