from loss.tuner import GoldTuner
from torch import Tensor

def fx(x: Tensor):
    return x[0] ** 2 + (x[1] - 2)**2


tuner = GoldTuner([5, 7], device="cpu", rule_eps=0.01, radius=0.5)
for i in range(500):
    x = tuner.get_coef()
    y = fx(x).float()
    # print("x:", x.numpy(), "y: ", y)
    tuner.update(y)

