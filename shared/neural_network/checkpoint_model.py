# checkpoint_model.py

import torch
import os.path

from shared import GPTLanguageModel

from internal.frame_stack.exceptions import MMError

def load_model(model_checkpoint_path, device, encoder):
    if os.path.exists(model_checkpoint_path):
        if device == 'cpu':
            checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_checkpoint_path)
        # model
        n_embd = checkpoint['n_embd']
        n_head = checkpoint['n_head']
        block_size = checkpoint['block_size']
        dropout = checkpoint['dropout']
        n_layer = checkpoint['n_layer']
        vocab_size = checkpoint['vocab_size']
        if vocab_size != len(encoder.tokens):
            raise Exception(f'vocab_size={vocab_size} is not equal to #encoder.tokens={len(encoder.tokens)}')
        model = GPTLanguageModel(n_embd, n_head, block_size, dropout, n_layer, device, encoder)
        model.epoch = checkpoint['epoch']
        model.step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer
        learning_rate = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        raise MMError(f'Model not found at path={os.path.abspath(model_checkpoint_path)}')
    model.to(device)
    return model, optimizer


def save_model(model, optimizer, model_checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'n_embd': model.n_embd,
        'n_head': model.n_head,
        'block_size': model.block_size,
        'dropout': model.dropout,
        'n_layer': model.n_layer,
        'device': model.device,
        'vocab_size': model.vocab_size,
        'epoch': model.epoch,
        'step': model.step
    }, model_checkpoint_path)
