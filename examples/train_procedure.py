import json

import torch
from torch import nn, Tensor

from dataset.probmeasure import UniformMeasure2D01, ProbabilityMeasureFabric
from dataset.toheatmap import ToGaussHeatMap
from gan.loss.loss_base import Loss
from gan.noise.stylegan import mixing_noise
from loss.hmloss import coord_hm_loss
from loss.losses import Samples_Loss
from metrics.writers import WR


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def gan_trainer(model, generator, decoder, encoder_HG, style_encoder, R_s, style_opt, g_transforms_1):

    def gan_train(i, real_img, condition):

        B = real_img.shape[0]
        C = 512

        requires_grad(generator, True)
        condition = condition.detach().requires_grad_(True)

        noise = mixing_noise(B, C, 0.9, real_img.device)

        fake, _ = generator(condition, noise, return_latents=False)

        model.discriminator_train([real_img], [fake], [condition])

        WR.writable("Generator loss", model.generator_loss)([real_img], [fake], [condition]) \
            .minimize_step(model.optimizer.opt_min)

        fake = fake.detach()
        trans_dict = g_transforms_1(image=fake, mask=condition)
        trans_fake_img = trans_dict["image"]
        trans_condition = trans_dict["mask"]

        trans_fake, fake_latent = generator(trans_condition, noise, return_latents=True)

        fake_latent_test = fake_latent[:, [0, 13], :].detach()
        fake_latent_pred = style_encoder(trans_fake.detach())

        restored = decoder(condition, style_encoder(real_img))

        coefs = json.load(open("../parameters/gan_loss.json"))

        (
            WR.L1("L1 fake")(trans_fake, trans_fake_img) * coefs["L1 fake"] +
            WR.L1("L1 restored")(restored, real_img) * coefs["L1 restored"] +
            WR.L1("L1 style gan")(fake_latent_pred, fake_latent_test) * coefs["L1 style gan"] +
            WR.L1("R_s")(style_encoder(fake), fake_latent_test) * coefs["R_s"]
        ).minimize_step(
            model.optimizer.opt_min,
            style_opt
        )

    return gan_train


def train_content(cont_opt, R_b, R_t, real_img, model, encoder_HG, decoder, generator, style_encoder):

    B = real_img.shape[0]
    C = 512

    heatmapper = ToGaussHeatMap(256, 1)

    requires_grad(encoder_HG, True)

    coefs = json.load(open("../parameters/content_loss.json"))
    encoded = encoder_HG(real_img)
    pred_measures: UniformMeasure2D01 = UniformMeasure2D01(encoded["coords"])

    heatmap_content = heatmapper.forward(encoded["coords"]).detach()


    ll = (
        WR.writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"] +
        WR.writable("R_t", R_t.__call__)(real_img, heatmap_content) * coefs["R_t"]
    )

    ll.minimize_step(cont_opt)


def content_trainer_with_gan(cont_opt, tuner, encoder_HG, R_b, R_t, model, generator, g_transforms, decoder, style_encoder):
    C = 512
    heatmapper = ToGaussHeatMap(256, 1)

    def do_train(real_img):

        B = real_img.shape[0]

        requires_grad(encoder_HG, True)
        requires_grad(decoder, False)

        coefs = json.load(open("../parameters/content_loss.json"))
        encoded = encoder_HG(real_img)
        pred_measures: UniformMeasure2D01 = UniformMeasure2D01(encoded["coords"])

        heatmap_content = heatmapper.forward(encoded["coords"]).detach()

        restored = decoder(encoded["skeleton"], style_encoder(real_img))

        noise = mixing_noise(B, C, 0.9, real_img.device)
        fake, _ = generator(encoded["skeleton"], noise)
        fake_content = encoder_HG(fake.detach())["coords"]

        ll = (
                WR.writable("R_b", R_b.__call__)(real_img, pred_measures) * coefs["R_b"] +
                WR.writable("R_t", R_t.__call__)(real_img, heatmap_content) * coefs["R_t"] +
                WR.L1("L1 image")(restored, real_img) * coefs["L1 image"] +
                WR.writable("fake_content loss", coord_hm_loss)(
                    fake_content, heatmap_content
                ) * coefs["fake_content loss"] +
                WR.writable("Fake-content D", model.loss.generator_loss)(
                    real=None,
                    fake=[fake, encoded["skeleton"].detach()]) * coefs["Fake-content D"]
        )

        ll.minimize_step(cont_opt)



    return do_train


def sup_loss(pred_mes, target_mes):

    heatmapper = ToGaussHeatMap(256, 1)

    pred_hm = heatmapper.forward(pred_mes.coord)
    pred_hm = pred_hm / (pred_hm.sum(dim=[2, 3], keepdim=True).detach() + 1e-8)
    target_hm = heatmapper.forward(target_mes.coord).detach()
    target_hm = target_hm / target_hm.sum(dim=[2, 3], keepdim=True).detach()

    return Loss(
        nn.BCELoss()(pred_hm, target_hm) * 100 +
        nn.MSELoss()(pred_mes.coord, target_mes.coord) * 0.5
    )


def content_trainer_supervised(cont_opt, encoder_HG, loader):
    def do_train():

        requires_grad(encoder_HG, True)
        w300_batch = next(loader)
        w300_image = w300_batch['data'].cuda()
        landmarks = w300_batch["meta"]["keypts_normalized"].cuda()
        w300_mes = UniformMeasure2D01(torch.clamp(landmarks, max=1))
        pred_coord = encoder_HG(w300_image)["coords"]
        pred_mes = UniformMeasure2D01(pred_coord)

        coefs = json.load(open("../parameters/content_loss.json"))

        WR.writable("W300 Loss", sup_loss)(pred_mes, w300_mes).__mul__(coefs["borj4_w300"])\
            .minimize_step(cont_opt)

    return do_train