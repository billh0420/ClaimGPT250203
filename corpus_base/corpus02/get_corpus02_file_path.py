# from corpus_base.corpus02
# get_corpus02_file_path.py

import pathlib

def get_corpus02_file_path():
    corpus02_file_path = pathlib.Path(__file__).parent.joinpath('corpus02.txt').resolve()
    return corpus02_file_path
