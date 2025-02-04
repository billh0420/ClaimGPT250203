# from shared.classes.trainer
# trainer.py

import torch
import os

from torch.utils.data import DataLoader, Subset

from shared.trainer.step_logger import StepLogger
from shared.trainer.sample_dataset import SampleDataset
from shared.neural_network.checkpoint_model import save_model

class Trainer:

    def __init__(self, model, optimizer, model_checkpoint_path):
        self.model = model
        self.optimizer = optimizer
        self.model_checkpoint_path = model_checkpoint_path

    def train_and_evaluate(self, max_train_epochs: int, batch_size: int, eval_interval: int, train_dataset, evaluator, step_logger: StepLogger or None):
        loss = None
        for epoch_step in range(max_train_epochs):
            self.model.train()
            data_loader = get_subset_dataloader(dataset=train_dataset, batch_size=batch_size, device=self.model.device)
            for x, y in iter(data_loader):
                # forward pass
                logits, loss = self.model.forward(x, y)
                # backward pass
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # update
                self.optimizer.step()
                self.model.step += batch_size
                if step_logger is not None:
                    if self.model.step % (100 * batch_size) == 0:
                        step_logger.log_step(step=self.model.step, max_examples=10)
            self.model.epoch += 1
            # every once in a while evaluate the loss on validation set
            if epoch_step % eval_interval == 0 or epoch_step == max_train_epochs - 1:
                if evaluator is not None:
                    mean_loss = evaluator.estimate_loss(self.model)
                    print(f'{epoch_step:8}. val loss {mean_loss:.4f}')
                else:
                    if loss is not None:
                        print(f'{epoch_step:8}. train loss {loss.detach().item():.4f}')
            if 1 < epoch_step < max_train_epochs and (epoch_step * batch_size) % 5000 == 0:
                print(f'saving model: epoch_step={epoch_step} model_checkpoint_path={os.path.abspath(self.model_checkpoint_path)}')
                save_model(self.model, self.optimizer, self.model_checkpoint_path)

def get_subset_dataloader(dataset: SampleDataset, batch_size: int, device):  # Note: 240908 SampleDataSet should be SizedDataset (a Dataset with __len__)
    # Define the desired subset size
    subset_size = 1 * batch_size
    # Create a subset of the dataset
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, subset_indices)
    # Create the subset DataLoader
    subset_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device=device))
    return subset_dataloader
