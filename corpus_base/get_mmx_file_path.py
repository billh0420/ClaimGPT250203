# from corpus_base
# mmx_file_path.py

import pathlib

def get_mmx_file_path():
    mmx_file_path = pathlib.Path(__file__).parent.joinpath('set.new2023.mmx').resolve()
    return mmx_file_path
