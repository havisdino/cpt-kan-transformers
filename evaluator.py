import torch
import torch.distributed as dist
from torchmetrics.functional.text.perplexity import perplexity


class Evaluator:
    def __init__(self, model, ignore_index):
        self.model = model
        self.ignore_index = ignore_index
    
    @torch.no_grad()
    def get_perplexity(self, input_ids, target_ids):
        self.model.eval()
        with torch.autocast('cuda', torch.float16):
            logits = self.model(input_ids)
        ppl = perplexity(logits, target_ids, ignore_index=self.ignore_index)
        return ppl

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        ppls = []
        
        for input_ids, target_ids in data_loader:
            input_ids = input_ids.to(dist.get_rank())
            target_ids = target_ids.to(dist.get_rank())
            
            with torch.autocast('cuda', torch.float16):
                ppls.append(self.get_perplexity(input_ids, target_ids))
        
        ppl = sum(ppls) / len(ppls)
        return ppl.item()
