# coding: utf-8

'''
Created by JamesYi
Created on 2018/10/23
'''

import argparse
import torch.nn.functional as F
from torch import optim
import torch
from torch.nn.utils import clip_grad_norm
from model import Encoder, AttentionDecoder, Seq2Seq
from utils import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100, help='Number of epochs for training')
    p.add_argument('-batch_size', type=int, default=32, help='Number of batch size for training')
    p.add_argument('-lr', type=float, default=0.0001, help='Initial learning rate for training')
    p.add_argument('-grad_clip', type=float, default=10.0, help='In case of gradient explosion')
    return p.parse_args()


def train(model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for index, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = src.cuda()
        trg = trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1), ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if index % 100 == 0 and index != 0:
            total_loss = total_loss / 100
            print("{} [Loss: %5.2f]".format(index, total_loss))
            total_loss = 0


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for index, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = src.cuda()
        trg = trg.cuda()
        output = model(src, trg, teacher_forcing_ratio=0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1), ignore_index=pad)
        total_loss += loss.item()
        return total_loss / len(val_iter)


if __name__ == "__main__":
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print('Loading dataset ......')
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print()

