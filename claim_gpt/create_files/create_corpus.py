# create_corpus.py
# from claim_gpt

from pathlib import Path

from shared import Encoder
from shared import get_tokens

def create_corpus(max_count: int, block_size: int, corpus_file_path: Path):
    print(f'Start')
    corpus_folder_path = corpus_file_path.parent
    corpus_statements = _get_corpus_statements(max_count, block_size, corpus_file_path)
    with open(corpus_file_path, 'w') as file:
        for statement in corpus_statements:
            file.write(f'{statement}\n')

    # save Encoder as json file (at least itos portion)
    print(f'create_encoder: corpus_folder_path={corpus_folder_path}')
    tokens = get_tokens(corpus_statements)
    encoder = Encoder(tokens=tokens)
    encoder.save_to_json(corpus_folder_path=corpus_folder_path)
    print(f'vocab_size={len(encoder.tokens)}')

    # Summary
    print(f'#corpus_statements={len(corpus_statements)}')
    print(f'Done')

def _get_corpus_statements(max_count, block_size, corpus_file_path: Path):
    claim_corpus_file_name = 'claim_corpus.txt'
    claim_corpus_file_path = corpus_file_path.parent.joinpath(claim_corpus_file_name)
    all_statements = _get_statements(file_name=claim_corpus_file_path)
    corpus_statements = []
    corpus_statement_count = 0
    prefixes = ['ax-mp ', 'mp2 ', 'mp2b ', 'mpd ', 'syl ']
    counts_by_prefix = dict(zip(prefixes, [0] * len(prefixes)))
    for statement in all_statements:
        corpus_statement = None
        for prefix in prefixes:
            if statement.startswith(prefix):
                corpus_statement = statement[len(prefix):]
                counts_by_prefix[prefix] += 1
                break
        if corpus_statement is None:
            continue
        if corpus_statement.startswith('<|start_claim|> <|conclude|>'):
            continue
        split_corpus_statement = corpus_statement.split()
        if len(split_corpus_statement) > block_size:
            continue
        corpus_statements.append(corpus_statement)
        corpus_statement_count += 1
        if corpus_statement_count >= max_count:
            break
    print(f'last corpus_statement: {corpus_statements[-1]}')
    for prefix in prefixes:
        print(f'{prefix[:-1]}-count={counts_by_prefix[prefix]}')
    return corpus_statements

def _get_statements(file_name, max_proved_statement_count=None):
    file = open(file_name, 'r')
    proved_statement_count = 0
    statements = []
    while True:
        statement = file.readline().rstrip()
        if not statement:
            break
        statements.append(statement)
        if statement.startswith('$p'):
            tau = statement.split(" ", maxsplit=3)
            assert len(tau) == 4
            if tau[2] == '|-':
                proved_statement_count += 1
        if max_proved_statement_count is not None and proved_statement_count >= max_proved_statement_count:
            break
    file.close()
    return statements
