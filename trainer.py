from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.text.perplexity import perplexity

from logger import TensorBoardLogger
from utils import save_checkpoint


@dataclass
class Trainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.cuda.amp.GradScaler
    grad_accum_interval: int
    ckp_retention: int
    ckp_interval: int
    
    def __post_init__(self):
        if dist.get_rank() == 0:
            self.logger = TensorBoardLogger('logs')
        self.epoch = 1
    
    def get_loss(self, input_ids, target_ids):
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        self.logits = logits.detach()
        return loss
    
    def train_step(self, input_ids, target_ids):
        self.model.train()
        with torch.autocast('cuda', torch.float16):
            self.loss = self.get_loss(input_ids, target_ids)
            self.loss /= self.grad_accum_interval
        self.scaler.scale(self.loss).backward()
        
    def accumulate_gradient(self):
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
    
    @property        
    def batch_loss(self):
        return self.loss.detach().item() * self.grad_accum_interval
        
    def fit(self, train_loader, n_steps):
        if dist.get_rank() == 0:
            self.logger.set_n_steps(n_steps)
            print(f'Accumulating gradients after {self.grad_accum_interval} substeps')
        
        data_iter = iter(train_loader)
        self.optimizer.zero_grad()
        
        for step in range(1, n_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(train_loader)
                batch = next(data_iter) 
                
            batch = [x.to(dist.get_rank()) for x in batch]
            input_ids, target_ids = batch

            self.train_step(input_ids, target_ids)

            if step % self.grad_accum_interval == 0:
                self.accumulate_gradient()

                if dist.get_rank() == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    ppl = perplexity(self.logits, target_ids).item()
                    self.logger.log(self.epoch, loss=self.batch_loss, lr=lr, ppl=ppl)

                    if step % self.ckp_interval == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scaler,
                            self.lr_scheduler, step, self.ckp_interval,
                            self.ckp_retention
                        )
