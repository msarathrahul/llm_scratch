import torch
import urllib.request

def download_dataset(url, file_name):
    urllib.request.urlretrieve(url, file_name)

def build_vocab(text):
    vocab = sorted(list(set(text)))
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}
    return vocab, itos, stoi

def encode(s, stoi):
    return [stoi[ch] for ch in s]

def decode(l, itos):
    return ''.join([itos[i] for i in l])

def get_batches(data, split, batch_size, context_window):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]
    batch_data = {"train": train, "val": val, "test": test}[split]
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y
