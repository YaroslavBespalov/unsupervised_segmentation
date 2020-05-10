from itertools import chain

from torch import nn, Tensor
import torch

from gan.loss_base import Loss

from loss.tuner import CoefTuner, GoldTuner

x = Tensor([1, 6])[None, ...]
# pred = torch.tensor([[15., 16.]], requires_grad=True, device='cpu')
# print(pred.shape)

# x = torch.tensor([0.], requires_grad=True, device='cuda')
# optimizer = torch.optim.Adam([x])
# pred_opt = torch.optim.Adam([pred], lr=0.001)
y = Tensor([2, 12])[None, ...]
z = Tensor([4, 24])[None, ...]
# Tuner = CoefTuner([1.0, 1.0], x.device)
Tuner = GoldTuner([1.0, 1.0], x.device, rule_eps=0.01, radius=3)

class Y(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Parameter(torch.tensor([10.0, 15.0])[None, ...])

    def forward(self, *input):
        return self.y

y_module = Y()
y_opt = torch.optim.Adam(y_module.parameters(), lr=1e-3)

for i in range(30000):
    for j in range(100):
        pred = y_module(None)
        Loss1 = Loss(torch.sum((z - pred) ** 2))
        Loss2 = Loss(torch.sum((x - pred) ** 2))
        Tuner.sum_losses([Loss1, Loss2]).minimize_step(y_opt)
    for j in range(100):
        pred = y_module(None)
        # Loss1 = Loss(torch.sum((z - pred) ** 2))
        # Loss2 = Loss(torch.sum((x - pred) ** 2))
        igor = (pred - y).pow(2).sum().item()
        Tuner.update(igor)
        # Tuner.tune_module(None, y_module, [Loss1, Loss2], lambda a: Loss((a - y).pow(2).sum()), 0.001)
        # Tuner.tune(
        #     pred,
        #     lambda a, g: Loss((a - 0.001 * g - y).pow(2).sum()),
        #     [Loss1, Loss2])
    print("coefficients: ", Tuner.coefs.detach().numpy())
    print("pred: ", y_module(None).detach().numpy())

