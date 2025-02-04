# from shared.neural_network
# generate_predicted_dictum.py

import torch

from shared.neural_network.generate_tokens import generate_tokens

def generate_predicted_dictum(prompt: str, terminal_token: str, model) -> str:
    assert isinstance(terminal_token, str)
    encoder = model.encoder
    split_prompt = prompt.split()
    max_new_tokens = model.block_size - len(split_prompt)
    encoded_prefix = torch.tensor([encoder.encode(prompt.rstrip())])
    terminal_token_id = encoder.stoi[terminal_token]
    generated_tokens = generate_tokens(max_new_tokens, encoded_prefix, model=model, terminal_token_id=terminal_token_id)[0].tolist()
    predicted_dictum = encoder.remove_trailing_space_tokens(encoder.decode(generated_tokens))
    return predicted_dictum
