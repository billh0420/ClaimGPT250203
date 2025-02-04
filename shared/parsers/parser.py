# from shared.parsers
# parser.py

from pathlib import Path

from shared.tokens.tokens import Tokens

class Parser:

    def __init__(self, mm_file_path: Path):
        self.tokens = Tokens(file_path=mm_file_path)
