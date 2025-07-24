import torch # for W2 calculation purpose
try:
    from geomloss import SamplesLoss
except:
    #pip3 install geomloss
    from geomloss import SamplesLoss
import numpy as np

def W2(X, Y):
    X = torch.from_numpy(X).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32)

    return SamplesLoss(loss='sinkhorn', p=2)(X, Y).numpy()

