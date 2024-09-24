import math
from torch import nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, eta_min=1e-6
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == "__main__":
    model = nn.Linear(512, 256)
    optimizer = Adam(model.parameters(), lr=2e-4)
    num_warmup_steps = int(np.floor(5000 / 1))
    num_training_steps = int(np.floor(5000/ 1)) * 300
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    lrs = []
    for i in tqdm(range(num_training_steps)):
        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

    sns.lineplot(x=range(len(lrs)), y=lrs)
    plt.savefig('lr.png')
    
    