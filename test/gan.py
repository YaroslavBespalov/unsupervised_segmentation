import unittest
import torch
import copy
from torch import optim
import torch.distributions as tdist
from framework.gan.euclidean.generator import EGenerator
from framework.gan.euclidean.discriminator import EDiscriminator
from framework.gan.gan_model import GANModel
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.vanilla import DCGANLoss
from framework.gan.loss.wasserstein import WassersteinLoss
import numpy as np

np.random.seed(42)

noise = torch.randn(100, 2)
target = torch.randn(100, 2) * 2

netG = EGenerator(size=2)
netD = EDiscriminator(dim=2)

fake = netG(noise)

hinge_gan_model = GANModel(netD, HingeLoss(netD), lr=0.0002)
was_gan_model = GANModel(netD, WassersteinLoss(netD, lambda x,y: (x+y)/2, penalty_weight=10), lr=0.0002)
vanilla_gan_model = GANModel(netD, DCGANLoss(netD), lr=0.0002)


class TestGANModel(unittest.TestCase):

    def test_hinge(self):

        loss_d = (torch.min(torch.zeros(100), -1 + netD(target))).mean() + (torch.min(torch.zeros(100), -1 - netD(fake))).mean()
        loss_g = -netD(fake).mean()
        loss_d_1 = hinge_gan_model.discriminator_loss(target, fake)
        loss_g1 = hinge_gan_model.generator_loss(target, fake)
        self.assertAlmostEqual(loss_d.item(), loss_d_1.item(), delta=1e-5)
        self.assertAlmostEqual(loss_g.item(), loss_g1.item(), delta=1e-5)

    def test_was(self):
        fake = netG(noise)
        x0 = (copy.deepcopy(target) + fake)/2
        x0.requires_grad_(True)
        loss_g = -netD(fake).mean()
        grad = torch.autograd.grad(netD(x0).sum(), x0)[0]
        loss_w = netD(target).mean() - netD(fake).mean() - 10*((grad.norm(2, dim=1) - 1) ** 2).mean()
        loss_w_1 = was_gan_model.discriminator_loss(target, fake)
        loss_g1 = was_gan_model.generator_loss(target, fake)

        self.assertAlmostEqual(loss_w.item(), loss_w_1.item(), delta=1e-5)
        self.assertAlmostEqual(loss_g.item(), loss_g1.item(), delta=1e-5)


    def test_van(self):
        loss_g = -torch.log(netD(fake).sigmoid()).mean()
        loss_v = (torch.log(netD(target).sigmoid())).mean() + torch.log(torch.ones_like(netD(fake)) - netD(fake).sigmoid()).mean()
        loss_v_1 = vanilla_gan_model.discriminator_loss(target, fake)
        loss_g1 = vanilla_gan_model.generator_loss(target, fake)
        self.assertAlmostEqual(loss_v.item(), loss_v_1.item(), delta=1e-5)
        self.assertAlmostEqual(loss_g.item(), loss_g1.item(), delta=1e-5)


