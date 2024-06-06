import os

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_train_loader
from evaluator import Evaluator
from lr_scheduler import WarmUpLR
from model import KANGPTLMHeadModel
from trainer import Trainer
from utils import Config


def setup(rank, master_addr, master_port, device_ids):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    
    dist.init_process_group('nccl', rank=rank, world_size=len(device_ids))


def main(rank, world_size, config):
    setup(
        rank, config.distributed.master_addr,
        config.distributed.master_port,
        config.distributed.device_ids
    )
    
    model = KANGPTLMHeadModel(config.kangpt)
    model.load_state_dict(torch.load(config.train.pretrain_path, 'cpu'))
    model = model.to(rank)
    model = DDP(model, [rank], rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = WarmUpLR(optimizer, **vars(config.train.lr))
    
    tokenizer = Tokenizer.from_file('pretrained/tokenizer.json')

    trainer = Trainer(
        model, optimizer, lr_scheduler, scaler,
        evaluator=Evaluator(model, tokenizer.encode('<|endoftext|>').ids[0]),
        grad_accum_interval=config.train.grad_accum_interval,
        ckp_retention=config.train.ckp_retention,
        ckp_interval=config.train.ckp_interval
    )
    
    train_loader = get_train_loader(
        rank, world_size, config.data.train_paths,
        config.data.n_tokens, config.train.batch_size, tokenizer
    )
    
    trainer.fit(train_loader, config.train.n_steps)
    dist.destroy_process_group()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    
    config = Config.from_yaml('config.yml')
    world_size = len(config.distributed.device_ids)
    mp.spawn(
        main,
        args=(world_size, config),
        nprocs=world_size
    )
