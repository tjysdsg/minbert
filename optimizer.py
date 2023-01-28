import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lamb = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            correct_bias = group['correct_bias']
            alpha = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                if 'm' not in state:  # 1st moment
                    state['m'] = torch.zeros_like(p)
                if 'v' not in state:  # 2nd moment
                    state['v'] = torch.zeros_like(p)
                if 't' not in state:  # time step
                    state['t'] = 0

                m = state['m']
                v = state['v']
                t = state['t'] + 1

                theta = p.data
                g = grad

                # Update first and second moments of the gradients
                # m = beta1 * m + g - g * beta1
                # v = beta2 * v + torch.square(g) - torch.square(g) * beta2
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * torch.square(g)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                a = alpha
                if correct_bias:
                    a = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Update parameters
                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data = theta - a * m / (torch.sqrt(v) + eps) - alpha * lamb * theta

                state['m'] = m
                state['v'] = v
                state['t'] = t

        return loss
