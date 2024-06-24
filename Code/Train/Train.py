import torch
import math

def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
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

def train_epoch(epoch, model, raw_model, train_loader, optimizer, scaler,
                learning_rate = 3e-4, decay_lr = None, 
                gradient_accumulation_steps = 1, grad_clip = 1.0,
                device = 'cuda'):
    
    iter_per_epoch=len(train_loader)
    for step, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)

        if decay_lr is None:
            lr = get_lr(epoch*iter_per_epoch+step)
        else:
            lr = learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.cuda.amp.autocast():
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
       