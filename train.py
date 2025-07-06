import torch
import numpy as np
import pandas as pd
import time

def evaluate_loss(model, dataset, get_batches, config):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

def train(model, optimizer, dataset, get_batches, config, scheduler=None, print_logs=False):
    losses = []
    start_time = time.time()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, dataset, get_batches, config)
            losses += [x]
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            start_time = time.time()
    print("Validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses)

def generate(model, decode, config, max_new_tokens=30):
    idx = torch.zeros(5, 1).long()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -config['context_window']:])
        last_time_step_logits = logits[:, -1, :]
        p = torch.nn.functional.softmax(last_time_step_logits, dim=-1)
        idx_next = torch.multinomial(p, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)
    return [decode(x) for x in idx.tolist()]
