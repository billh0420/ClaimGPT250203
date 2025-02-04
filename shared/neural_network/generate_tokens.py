# from shared.utility
# generate_tokens.py

import torch

from torch import Tensor

from torch.nn import functional as F

def generate_tokens(max_new_tokens, idx: Tensor, model, terminal_token_id: int or None) -> Tensor:
    # idx is (B, T) array of indices in the current context
    assert not model.training
    assert isinstance(terminal_token_id, int or None)
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -model.block_size:]  # at most block_size
        # get the predictions
        logits, loss = model.forward(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        if terminal_token_id is not None and idx_next.shape == (1, 1):
            if idx_next.cpu().numpy()[0][0] == terminal_token_id:
                break
    return idx
