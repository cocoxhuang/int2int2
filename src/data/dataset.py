from copy import deepcopy
import torch
from ..utils.config import load_config
import numpy as np
import random
import os
from src.data.tokenizer import BaseTokenizer


class Dataset:
    def __init__(self, data, tokenizer, base = 1000, seed = 42, batch_size = 32, eval_size = 10000):
        # Set random seeds FIRST for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For CUDA reproducibility
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.tokenizer = tokenizer
        self.dictionary = self.tokenizer._get_vocab()

        self.data = {}
        self.data['inputs'] = torch.tensor([self.tokenizer.tokenize(seq) for seq in data['input'].tolist()], dtype=torch.long)
        self.data['targets'] = torch.tensor([self.tokenizer.tokenize(seq) for seq in data['target'].tolist()], dtype=torch.long)

        self.eval_size = eval_size
        self.batch_size = batch_size
        self.train_dataloader, self.eval_dataloader = self.create_train_val_dataloader(eval_size=eval_size)

    def create_dataloader(self, data):
        dataset = torch.utils.data.TensorDataset(data['inputs'], data['targets'])
        # Create a generator with fixed seed for deterministic shuffling
        generator = torch.Generator()
        generator.manual_seed(self.seed)  # Fixed seed for reproducible shuffling
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator  # Use seeded generator for reproducible shuffle
        )
        return dataloader

    def create_train_val_dataloader(self, eval_size=10000):
        shuffle_indices = torch.randperm(len(self.data['inputs']), generator=torch.Generator().manual_seed(self.seed))
        train_idx = int(0.8 * len(self.data['inputs']))
        eval_idx = len(self.data['inputs']) - train_idx
        self.data['inputs'] = self.data['inputs'][shuffle_indices]
        self.data['targets'] = self.data['targets'][shuffle_indices]
        train_inputs = self.data['inputs'][:train_idx]
        train_targets = self.data['targets'][:train_idx]
        eval_inputs = self.data['inputs'][train_idx:]
        eval_targets = self.data['targets'][train_idx:]

        if eval_inputs.shape[0] > eval_size:
            eval_inputs = eval_inputs[:eval_size]
            eval_targets = eval_targets[:eval_size]

        train_dataloader = self.create_dataloader({'inputs': train_inputs, 'targets': train_targets})
        eval_dataloader = self.create_dataloader({'inputs': eval_inputs, 'targets': eval_targets})
        return train_dataloader, eval_dataloader