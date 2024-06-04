import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmUpLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_init: float,
        lr_peak: float,
        lr_min: float,
        warmup_step: int,
        decreasing_steepness: float
    ):
        def schedule(step):
            if step <= warmup_step:
                alpha = (lr_peak - lr_init) / (warmup_step ** 2)
                lr = alpha * (step ** 2) + lr_init
            else:
                beta = - warmup_step - math.log(lr_peak - lr_min) / decreasing_steepness
                lr = math.exp(-(step + beta) * decreasing_steepness) + lr_min
            return lr

        LambdaLR.__init__(self, optimizer, schedule)