import torch
import math
import logging
import torch.nn.functional as F

def get_lr(it, learning_rate = 3e-4, min_lr = 1e-5, warmup_iters = 1000, lr_decay_iters = 80000):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch, model, train_loader, optimizer, scaler,
                learning_rate = 3e-4, decay_lr = None, grad_clip = 1.0,
                device = 'cuda'):
    
    iter_per_epoch=len(train_loader)
    for step, (X, Y,loss_mask) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        loss_mask = loss_mask.to(device)

        if decay_lr is None:
            lr = get_lr(epoch*iter_per_epoch+step)
        else:
            lr = learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.cuda.amp.autocast():
            logits = model(X, Y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0,reduce=False)
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss*loss_mask)/loss_mask.sum()
        
        logging.info(f'step: {step}, loss: {loss.item(): .4f}')
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
       