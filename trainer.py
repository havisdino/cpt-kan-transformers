from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from evaluator import Evaluator
from logger import TensorBoardLogger
from utils import save_checkpoint


@dataclass
class Trainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.cuda.amp.GradScaler
    evaluator: Evaluator
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
    
    @torch.no_grad()
    def evaluate(self, input_ids, target_ids):
        if dist.get_rank() == 0:
            ppls = [None for _ in range(dist.get_world_size())]
        else:
            ppls = None
        
        self.model.eval()
        ppl = self.evaluator.get_perplexity(input_ids, target_ids)
        dist.all_gather_object(ppl, ppls)
        
        if dist.get_rank() == 0:
            self.gathered_ppl = sum(ppls) / len(ppls)

    @torch.no_grad()
    def _gather_batch_loss(self):
        if dist.get_rank() == 0:
            batch_losses = [None for _ in range(dist.get_world_size())]
        else:
            batch_losses = None

        dist.all_gather_object(self.batch_loss, batch_losses)

        if dist.get_rank() == 0:
            self.gathered_batch_loss = sum(batch_losses) / len(batch_losses)
        
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

            dist.barrier()
            if step % self.grad_accum_interval == 0:
                self.accumulate_gradient()

                dist.barrier()
                self.evaluate(input_ids, target_ids)
                self._gather_batch_loss()

                if dist.get_rank() == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log(self.epoch, train_loss=self.gathered_batch_loss, lr=lr, train_ppl=self.gathered_ppl)
                    print(f'\ttrain_ppl: {self.gathered_ppl}')

                if step % self.ckp_interval == 0 and dist.get_rank() == 0:
                    save_checkpoint(
                        self.model, self.optimizer, self.scaler,
                        self.lr_scheduler, step, self.ckp_interval,
                        self.ckp_retention
                    )
                dist.barrier()
                