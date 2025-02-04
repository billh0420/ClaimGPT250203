# from claim_gpt.create_files
# create_files.py

from pathlib import Path

from claim_gpt.create_files.create_corpus import create_corpus
from claim_gpt.create_files.create_claim_corpus import create_claim_corpus

def create_files(output_folder_path: Path, block_size: int, limit_count: int) -> Path:
    corpus_folder_path = output_folder_path.joinpath('corpus').resolve()
    corpus_file_path = corpus_folder_path.joinpath('corpus.txt').resolve()
    if not corpus_file_path.exists():
        create_claim_corpus(corpus_file_path=corpus_file_path)
        create_corpus(max_count=limit_count, block_size=block_size, corpus_file_path=corpus_file_path)
    return corpus_file_path
