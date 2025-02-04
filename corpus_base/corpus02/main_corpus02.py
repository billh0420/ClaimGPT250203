# main_corpus02.py

import random

from pathlib import Path

from corpus_base.corpus02.get_corpus02_file_path import get_corpus02_file_path
from shared import Parser02 as Parser

print("main_corpus02: Start parser02")

mmx_file_path = Path('../set.new2023.mmx').resolve()
parser = Parser(mmx_file_path=mmx_file_path)
parser.parse()

corpus02_file_path = get_corpus02_file_path()
with open(corpus02_file_path, 'w') as file:
    for line in parser.result:
        file.write(f'{line}\n')

print(f'count={parser.count}')
print(f'axiom_statement: {random.choice(parser.axiom_statements)}')
print(f'proved_statement: {random.choice(parser.proved_statements)}')

print(f'#proved_statement_labels={len(parser.proved_statement_labels)}')
print(f'#used_proved_statement_labels={len(parser.used_proved_statement_labels)}')

print("Stop")
