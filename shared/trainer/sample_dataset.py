# sample_dataset.py

import torch

from torch.utils.data import Dataset

from shared import Encoder


class SampleDataset(Dataset):

    def __init__(self, encoded_statements: list[list[int]], encoder: Encoder, device: str):
        # FIXME: 240902 pass in statements rather than encoded_statements ???
        self.encoder = encoder
        self.device = device
        self.samples = []
        self.space_token = self.encoder.stoi['@']
        self.encoded_statements = encoded_statements
        self.samples = [_get_sample(statement, self.encoder, self.device) for statement in self.encoded_statements]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx][0]
        label = self.samples[idx][1]
        return item, label


def _get_sample(statement: list[int], encoder: Encoder, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    space_token = encoder.space_token
    x = statement
    y = statement[1:] + [space_token]
    x, y = torch.tensor(x), torch.tensor(y)
    x, y = x.to(device), y.to(device)
    return x, y
