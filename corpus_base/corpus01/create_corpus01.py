# create_corpus01.py
# from corpus_base/corpus01

import random
from pathlib import Path
from corpus_base.get_mmx_file_path import get_mmx_file_path

def create_corpus01():
    corpus01_file_path = Path('corpus01.txt').resolve()
    if not corpus01_file_path.exists():
        from shared import Parser01 as Parser
        print("Begin create_corpus01: Start parser01")
        mmx_file_path = get_mmx_file_path()
        parser = Parser(mmx_file_path=mmx_file_path)
        parser.parse()
        with open(corpus01_file_path, 'w') as file:
            for line in parser.statements:
                file.write(f'{line}\n')
        print(f'count={parser.count}')
        print(f'axiom_statement: {random.choice(parser.axiom_statements)}')
        print(f'proved_statement: {random.choice(parser.proved_statements)}')
        print(f'#proved_statement_labels={len(parser.proved_statement_labels)}')
        print(f'#used_proved_statement_labels={len(parser.used_proved_statement_labels)}')
        print("Done create_corpus01")
    return corpus01_file_path
