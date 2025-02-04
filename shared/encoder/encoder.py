# encoder.py

import json

from pathlib import Path

class Encoder:

    def __init__(self, tokens):
        # create a mapping from token to integers and vice versa
        self.stoi = {token: i for i, token in enumerate(tokens)}
        self.itos = {i: token for i, token in enumerate(tokens)}

    @property
    def tokens(self) -> list[str]:
        return list(self.itos.values())

    @property
    def space_token(self) -> int:
        return self.stoi['@']

    # encoder: take a string s of tokens, output a list of integers
    def encode(self, s):
        return [self.stoi[token] for token in s.split()]

    # decoder: take a list ls of integers, output a string of tokens separated by spaces
    def decode(self, ls):
        return ' '.join([self.itos[i] for i in ls])

    def remove_trailing_space_tokens(self, statement: str) -> str:
        assert isinstance(statement, str)
        statement_size = len(statement)
        argumented_space_token = f' {self.itos[self.space_token]}'
        argumented_space_token_size = len(argumented_space_token)
        index = statement_size
        for _ in range(statement_size - 1):
            if statement[index - argumented_space_token_size: index] == argumented_space_token:
                index -= argumented_space_token_size
            else:
                break
        return statement[:index]

    def save_to_json(self, corpus_folder_path: Path):
        encoder_file_path = corpus_folder_path.joinpath('encoder.json').resolve()
        with encoder_file_path.open("w") as encoder_file:
            json.dump(self.itos, encoder_file, indent=0, sort_keys=True)

    @classmethod
    def load_from_json(cls, corpus_folder_path: Path):
        encoder_file = open(corpus_folder_path.joinpath('encoder.json'), 'r')
        itos = json.load(encoder_file)
        encoder_file.close()
        encoder = Encoder(tokens='')
        encoder.itos = {int(key): value for (key, value) in itos.items()}
        encoder.stoi = {value: key for (key, value) in encoder.itos.items()}
        return encoder
