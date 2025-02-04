# train_model.py
# from claim_gpt

import torch
import time
import os.path

from pathlib import Path

from shared import save_model
from shared import Trainer
from shared import SampleDataset
from shared import get_encoded_statement
from shared import get_n_layer

def train_model(model, optimizer, max_train_epochs: int, corpus_file_path: Path, model_file_path: Path):
    if max_train_epochs == 0:
        return model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get the encoder from tokens to integers and vice versa
    encoder = model.encoder

    # here are all the unique tokens that occur in this text
    vocab_size = len(encoder.tokens)
    print(f'vocab_size={vocab_size}')

    # get max_statement_count of statements
    print(f'corpus_file_path={os.path.abspath(corpus_file_path)}')
    with open(corpus_file_path, 'r') as file:
        corpus_statements = file.read().splitlines()
    corpus_statement_count = len(corpus_statements)
    print(f'corpus_statement_count={corpus_statement_count}')

    # get encoded_train_statements
    encoded_train_statements = []
    for i in range(corpus_statement_count):
        statement = corpus_statements[i]
        encoded_train_statement = get_encoded_statement(statement, encoder, model.block_size)
        encoded_train_statements.append(encoded_train_statement)
    print(f'#encoded_train_statements={len(encoded_train_statements)}')

    # create train_dataset
    train_dataset = SampleDataset(encoded_train_statements, encoder, device)

    # train model
    train_batch_size = 10 * 1  # how many independent sequences will we process in parallel?
    eval_interval = max(1, max_train_epochs // 10)  # show progress
    n_layer = get_n_layer(model=model)
    learning_rate = optimizer.defaults['lr']
    print(f'=== train and evalate model ===')
    print(f'epoch={model.epoch}; step={model.step}; n_head={model.n_head}; n_layer={model.n_layer}')
    print(f'train_batch_size={train_batch_size}')
    print(f'max_train_epochs={max_train_epochs} eval_interval={eval_interval}')
    print(f'block_size={model.block_size} n_layer={n_layer} learning_rate={learning_rate} device={device}')
    print(f'#train_dataset={len(train_dataset)}')
    start_time = time.time()
    trainer = Trainer(model, optimizer, model_file_path)
    trainer.train_and_evaluate(max_train_epochs, train_batch_size, eval_interval, train_dataset, evaluator=None, step_logger=None)
    elapsed_time = time.time() - start_time
    print(f'elapsed_time={elapsed_time / 60:.2f} minutes')
    print(f'epoch={model.epoch}; step={model.step}')
    # save model and optimizer
    print(f'saving model: epoch={model.epoch} model_checkpoint_path={os.path.abspath(model_file_path)}')
    save_model(model, optimizer, model_file_path)
