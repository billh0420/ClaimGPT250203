# get_encoded_statement.py

from shared import Encoder

def get_encoded_statement(statement: str, encoder: Encoder, block_size: int) -> list[int]:
    encoded_statement = encoder.encode(statement)
    count = len(encoded_statement)
    if count < block_size:
        space_token = encoder.space_token
        encoded_statement = encoded_statement + [space_token] * (block_size - count)
    elif count > block_size:
        encoded_statement = encoded_statement[0: block_size]
    return encoded_statement
