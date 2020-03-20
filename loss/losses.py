import torch.nn as nn
import torch
from geomloss.sinkhorn_samples import cost_routines, softmin_tensorized
from geomloss.utils import scal
from torch.nn.functional import softmin

from dataset.probmeasure import ProbabilityMeasure
from geomloss import SamplesLoss
from loss_base import Loss



class Samples_Loss(nn.Module):
    def __init__(self, blur=.01, scaling=.9, diameter=None):
        super(Samples_Loss, self).__init__()
        self.loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=False, diameter=diameter)
        self.pot = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=False, potentials=True)

    def forward(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):
        batch_loss = self.loss(m1.probability, m1.coord, m2.probability, m2.coord)
        return Loss(batch_loss.mean())

    def find_p(self, m1: ProbabilityMeasure, m2: ProbabilityMeasure):

        a, b = self.pot(m1.probability.type(torch.float64),
                        m1.coord.type(torch.float64),
                        m2.probability.type(torch.float64),
                        m2.coord.type(torch.float64))
        batch_size = m1.coord.shape[0]
        eps = 0.01**2

        C_xy = cost_routines[2](m1.coord.type(torch.float64), m2.coord.type(torch.float64))

        D = a.view(batch_size, -1, 1) + b.view(batch_size, 1, -1) - C_xy

        P = torch.exp(D / eps).type(torch.float32)

        return P




