from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class TensorBoardLogger(SummaryWriter):
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        SummaryWriter.__init__(self, log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        import logging
        logging.getLogger('tensorflow').disabled = True
        
        self.global_step = 1
        self.pbar = tqdm()
    
    def set_n_steps(self, n_steps):
        self.pbar.total = n_steps
    
    def log(self, epoch=None, **kwargs):        
        for tag, value in kwargs.items():
            self.add_scalar(tag, value, self.global_step)
        
        if epoch is not None:
            self.pbar.set_description(f'epoch {epoch}')
        self.pbar.set_postfix(**kwargs)
        self.pbar.update()    
        
        self.global_step += 1
    
    def close(self):
        SummaryWriter.close(self)
