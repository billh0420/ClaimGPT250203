# tokens.py

from typing import TextIO

from pathlib import Path

def get_tokens(statements):
    set_tokens = set().union(*[set(statement.split(' ')) for statement in statements])
    if '' in set_tokens:
        set_tokens.remove('')
    assert '' not in set_tokens
    assert ' ' not in set_tokens
    assert '@' not in set_tokens
    tokens = ['@'] + sorted(list(set_tokens))
    return tokens

class Tokens:

    def __init__(self, file_path: Path):
        self.token_buffer = []
        self.token_buffer_size = 0
        self.token_index = 0
        self.mm_file: TextIO = open(file_path, 'r')

    def close_file(self):  # Note this: kludge
        self.mm_file.close()

    def get_next_token(self):
        token = self._get_next_raw_token()
        # skip comments
        while token == '$(':
            while token and token != '$)':
                token = self._get_next_raw_token()
            if token:
                token = self._get_next_raw_token()
            else:
                raise Exception('Comment not closed')
        return token

    def _get_next_raw_token(self):
        token = None
        while token is None:
            if self.token_index < self.token_buffer_size - 1:
                self.token_index += 1
            else:
                self.token_buffer_size = 0
                self.token_index = 0
                line = self.mm_file.readline()
                if line:
                    self.token_buffer = line.split()
                    self.token_buffer_size = len(self.token_buffer)
                else:
                    self.close_file()
                    return None
            if self.token_buffer:  # Note: token_buffer can be empty when line is just whitespace
                token = self.token_buffer[self.token_index]
                return token
