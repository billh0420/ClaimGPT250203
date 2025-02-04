# from claim_gpt
# evaluate_model.py

import math

from pathlib import Path

from claim_gpt.validate_model.validate import validate
from shared import get_encoded_statement

def validate_model(model, max_examples: int, max_print_error: int, max_print_ok: int, corpus_file_path: Path):
    model.eval() # set to eval mode
    block_size = model.block_size
    encoder = model.encoder

    # get max_statement_count of statements
    print(f'corpus_file_path={corpus_file_path}')
    with open(corpus_file_path, 'r') as file:
        corpus_statements = file.read().splitlines()
    corpus_statement_count = len(corpus_statements)
    print(f'corpus_statement_count={corpus_statement_count}')

    # get statements
    encoded_train_statement_count = math.floor(corpus_statement_count / 1.25)
    encoded_val_statements = []
    for i in range(encoded_train_statement_count, corpus_statement_count):
        statement = corpus_statements[i]
        encoded_val_statement = get_encoded_statement(statement, encoder, block_size)
        encoded_val_statements.append(encoded_val_statement)

    # evaluate_model
    val_statements = [corpus_statements[x] for x in range(encoded_train_statement_count, corpus_statement_count)]
    print(f'epoch={model.epoch}; step={model.step}; n_head={model.n_head}; n_layer={model.n_layer}')
    print(f'#val_statements={len(val_statements)}')
    validate(statements=val_statements, max_examples=max_examples, encoder=encoder, model=model, block_size=block_size)
