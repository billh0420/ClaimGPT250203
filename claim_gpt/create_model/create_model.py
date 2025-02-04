# create_model.py
# from claim_gpt

import os
import torch

from pathlib import Path

from shared import GPTLanguageModel

from shared import save_model
from shared import Encoder

def create_model(model_file_path: Path, corpus_file_path: Path, n_head: int, n_layer: int):
    if not model_file_path.is_file():
        print(f'Start create model and optimizer')
        print(f'create_model: model_checkpoint_path={os.path.abspath(model_file_path)}')
        if os.path.exists(model_file_path):
            print(f'model already exist at path={os.path.abspath(model_file_path)}')
        else:
            _create_model(model_file_path, corpus_file_path=corpus_file_path, n_head=n_head, n_layer=n_layer)
            print(f'model created at path={os.path.abspath(model_file_path)}')
        print(f'Done')

def _create_model(model_checkpoint_path: Path, corpus_file_path: Path, n_head: int, n_layer: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_size = 150  # what is the maximum context length for predictions?
    learning_rate = 1e-4
    n_embd = 1000
    dropout = 0.2

    print(f'create_encoder')
    encoder = Encoder.load_from_json(corpus_folder_path=corpus_file_path.parent)
    print(f'vocab_size={len(encoder.tokens)}')

    # create model
    print(f'create model: block_size={block_size} device={device}')
    model = GPTLanguageModel(n_embd, n_head, block_size, dropout, n_layer, device, encoder)
    model = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    print('create a PyTorch optimizer')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # save model and optimizer
    save_model(model, optimizer, model_checkpoint_path)
