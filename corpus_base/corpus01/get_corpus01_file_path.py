# from corpus_base.corpus01
# get_corpus01_file_path.py

import pathlib

def get_corpus01_file_path():
    corpus01_file_path = pathlib.Path(__file__).parent.joinpath('corpus01.txt').resolve()
    return corpus01_file_path
