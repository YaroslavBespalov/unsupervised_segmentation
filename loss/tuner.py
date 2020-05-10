from typing import List, Callable, Union, Tuple, Optional

import torch
from torch import nn, Tensor

import copy
from gan.loss_base import Loss


class CoefTuner:

    def __init__(self, coefs: List[float], device):
        self.coefs = torch.tensor(coefs, dtype=torch.float32, device=device).requires_grad_(True)
        self.opt = torch.optim.Adam([self.coefs], lr=0.001)

    def opt_with_grads(self,  grads: List[Tensor], coef_loss: Callable[[Tensor], Loss]):

        grads_sum = 0

        for i, g in enumerate(grads):
            grads_sum = grads_sum + self.coefs[i] * g

        norm = grads_sum.pow(2).view(grads_sum.shape[0], -1).sum(1).sqrt() + 1e-5
        grads_sum = grads_sum / norm.view(-1, 1)
        coef_loss(grads_sum).minimize_step(self.opt)
        self.coefs.data = self.coefs.data.clamp_(0, 100)


    def tune(self,
             argument: Tensor,
             coef_loss: Callable[[Tensor, Tensor], Loss],
             losses: List[Union[Loss, Tuple[Loss, Tensor]]]) -> None:
        grads = []
        for loss in losses:
            tmp_arg = argument
            if isinstance(loss, tuple):
                tmp_arg = loss[1]
                loss = loss[0]
            g = torch.autograd.grad(loss.to_tensor(), [tmp_arg], only_inputs=True)[0].detach()
            grads.append(g)

        grads_sum = 0
        for i, g in enumerate(grads):
            grads_sum = grads_sum + self.coefs[i] * g

        norm = grads_sum.pow(2).view(argument.shape[0], -1).sum(1).sqrt() + 1e-5
        grads_sum = grads_sum / norm.view(-1, 1)
        coef_loss(argument, grads_sum).minimize_step(self.opt)
        self.coefs.data = self.coefs.data.clamp_(0, 100)

    def tune_module(self, input: Tensor, module: nn.Module, losses: List[Loss], module_loss: Callable[[Tensor], Loss],
                    lr: float):
        outputs = []

        out = module(input).detach()
        quality = module_loss(out).item()

        for loss in losses:
            module.zero_grad()
            loss.to_tensor().backward(retain_graph=True)
            module_new = copy.deepcopy(module)
            for p1, p2 in zip(module.parameters(), module_new.parameters()):
                p2._grad = p1._grad
                p2.data = p2.data - lr * p1._grad
            # module_new_opt = torch.optim.Adam(module_new.parameters(), lr=lr)
            # module_new_opt.step()
            q = module_new(input).detach()
            outputs.append(q)

        # self.opt_with_grads(grads, lambda g: module_loss(out + g))

        sum_outputs = 0
        norm = self.coefs.sum()
        for ind in range(len(outputs)):
            sum_outputs = sum_outputs + outputs[ind] * self.coefs[ind] / norm

        module_loss(sum_outputs).minimize_step(self.opt)
        self.coefs.data = self.coefs.data.clamp_(0, 100)

        return quality

    def sum_losses(self, losses: List[Loss]) -> Loss:
        res = Loss.ZERO()
        for i, l in enumerate(losses):
            res = res + l * self.coefs[i].detach()

        return res


class GoldTuner:
    def __init__(self, coefs: List[float], device, rule_eps: float, radius: float, active=True):
        self.coefs = torch.tensor(coefs, dtype=torch.float32, device=device)
        self.y: dict = {"y1": None, "y2": None}
        self.x1: Optional[Tensor] = None
        self.x2: Optional[Tensor] = None
        self.a: Optional[Tensor] = None
        self.b: Optional[Tensor] = None
        self.proector = lambda x: x.clamp_(0, 1000)
        self.radius = radius
        self.queue = []
        self.active = active
        if active:
            self.direction()
        else:
            self.queue = [("0", self.coefs)]
        self.rule_eps = rule_eps
        self.prev_opt_y = None
        self.directions_tested = 0
        self.direction_score = 100
        self.best_ab: Optional[Tuple[Tensor, Tensor]] = None

    def update_coords(self):
        self.x1 = self.a + 0.382 * (self.b - self.a)
        self.x2 = self.b - 0.382 * (self.b - self.a)
        self.queue.append(("1", self.x1))
        self.queue.append(("2", self.x2))

    def direction(self):
        assert len(self.queue) == 0
        rand_direction = torch.randn_like(self.coefs)
        rand_direction_norm = torch.sqrt((rand_direction ** 2).sum())
        rand_direction = rand_direction / rand_direction_norm
        self.a = self.coefs
        self.b = self.proector(self.a + self.radius * rand_direction)
        print(f"Choose direction from {self.a} to {self.b}")
        self.update_coords()

    def get_coef(self):
        return self.queue[0][1]

    def update(self, y: float):

        if not self.active:
            return None

        self.y["y" + self.queue[0][0]] = y
        self.queue.pop(0)
        if len(self.queue) == 0:

            if self.prev_opt_y and self.y["y1"] > self.prev_opt_y and self.y["y2"] > self.prev_opt_y and self.directions_tested < 5:
                print("Take new direction")
                self.directions_tested += 1

                score = min(self.y["y1"], self.y["y2"])
                if self.direction_score > score:
                    self.best_ab = (self.a, self.b)
                    self.direction_score = score

                self.direction()
                return None

            elif self.directions_tested >= 5:
                print(f"Take best from 5 direction {self.a} to {self.b}")
                self.a = self.best_ab[0]
                self.b = self.best_ab[1]
                self.update_coords()
                self.prev_opt_y = 100
                self.directions_tested = 0
                self.direction_score = 100
                self.best_ab = None
                return None

            self.directions_tested = 0
            self.direction_score = 100
            self.best_ab = None

            if self.stop_rule():
                print("opt val:", self.prev_opt_y)
                self.direction()
                return None

            self.optimize()


    def optimize(self):
        print(self.y)
        if self.y["y1"] >= self.y["y2"]:
            self.a = self.x1
            self.x1 = self.x2
            self.x2 = self.b - 0.382 * (self.b - self.a)
            self.queue.append(("2", self.x2))
            self.y["y1"] = self.y["y2"]
            print("new a: ", self.a)
        else:
            self.b = self.x2
            self.x2 = self.x1
            self.x1 = self.a + 0.382 * (self.b - self.a)
            self.queue.append(("1", self.x1))
            self.y["y2"] = self.y["y1"]
            print("new b: ", self.b)

    def stop_rule(self):
        if (self.b - self.a).abs().max() < self.rule_eps:
            self.coefs = (self.a + self.b) / 2
            self.prev_opt_y = (self.y["y1"] + self.y["y2"]) / 2
            self.x1 = None
            self.x2 = None
            self.queue = []
            print("coefs: ", self.coefs)
            return True
        return False

    def sum_losses(self, losses: List[Loss]) -> Loss:
        res = Loss.ZERO()
        coef = self.get_coef()
        for i, l in enumerate(losses):
            res = res + l * coef[i].detach()

        return res



