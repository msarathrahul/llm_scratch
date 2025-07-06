import torch
from config import MASTER_CONFIG
from data import download_dataset, build_vocab, encode, decode, get_batches
from model import Llama
from train import train, generate

def main():
    # Download and prepare data
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_name = "tinyshakespeare.txt"
    download_dataset(url, file_name)
    lines = open(file_name, 'r').read()
    vocab, itos, stoi = build_vocab(lines)
    MASTER_CONFIG['vocab_size'] = len(vocab)
    dataset = torch.tensor(encode(lines, stoi), dtype=torch.int8)

    # Build model
    model = Llama(MASTER_CONFIG)
    optimizer = torch.optim.Adam(model.parameters())

    # Train model
    train(model, optimizer, dataset, get_batches, MASTER_CONFIG)

    # Generate text
    samples = generate(model, lambda l: decode(l, itos), MASTER_CONFIG, max_new_tokens=500)
    print(samples[0])

if __name__ == "__main__":
    main()
