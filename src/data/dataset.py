import os
import random

import numpy as np
import torch


class Dataset:
    """Tokenised (input, target) pairs plus the 80/20 train/validation split.

    Tokenisation runs in chunks and writes straight into numpy arrays: the full
    corpus is ~2.1M sequences, and materialising that many Python lists at once
    needs several GB.
    """

    CHUNK = 100_000

    def __init__(self, data, tokenizer, base=1000, seed=42, batch_size=32, eval_size=10000):
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

        pad_id = self.tokenizer.vocab['<PAD>']
        self.data = {
            'inputs': self._encode(data['input'].tolist(), pad_id),
            'targets': self._encode(data['target'].tolist(), pad_id),
        }

        self.eval_size = eval_size
        self.batch_size = batch_size
        self.train_dataloader, self.eval_dataloader = self.create_train_val_dataloader(
            eval_size=eval_size)

    def _encode(self, sequences, pad_id):
        """Tokenise `sequences` and right-pad them to a common length."""
        chunks, width = [], 0
        for start in range(0, len(sequences), self.CHUNK):
            tokens = [self.tokenizer.tokenize(s) for s in sequences[start:start + self.CHUNK]]
            w = max(len(t) for t in tokens)
            block = np.full((len(tokens), w), pad_id, dtype=np.int64)
            for i, t in enumerate(tokens):
                block[i, :len(t)] = t
            chunks.append(block)
            width = max(width, w)
            del tokens

        out = np.full((len(sequences), width), pad_id, dtype=np.int64)
        row = 0
        for block in chunks:
            out[row:row + len(block), :block.shape[1]] = block
            row += len(block)
        chunks.clear()
        return torch.from_numpy(out)

    def _pad_sequences(self, sequences, pad_id):
        """Pad a list of token lists to a common length (kept for external callers)."""
        width = max(len(s) for s in sequences)
        return torch.tensor([s + [pad_id] * (width - len(s)) for s in sequences],
                            dtype=torch.long)

    def create_dataloader(self, data, shuffle=True):
        dataset = torch.utils.data.TensorDataset(data['inputs'], data['targets'])
        # Create a generator with fixed seed for deterministic shuffling
        generator = torch.Generator()
        generator.manual_seed(self.seed)  # Fixed seed for reproducible shuffling
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,  # Use seeded generator for reproducible shuffle
        )

    def create_train_val_dataloader(self, eval_size=10000):
        n = len(self.data['inputs'])
        shuffle_indices = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        train_idx = int(0.8 * n)
        self.data['inputs'] = self.data['inputs'][shuffle_indices]
        self.data['targets'] = self.data['targets'][shuffle_indices]

        train_inputs = self.data['inputs'][:train_idx]
        train_targets = self.data['targets'][:train_idx]
        eval_inputs = self.data['inputs'][train_idx:]
        eval_targets = self.data['targets'][train_idx:]

        if eval_inputs.shape[0] > eval_size:
            eval_inputs = eval_inputs[:eval_size]
            eval_targets = eval_targets[:eval_size]

        train_dataloader = self.create_dataloader(
            {'inputs': train_inputs, 'targets': train_targets})
        eval_dataloader = self.create_dataloader(
            {'inputs': eval_inputs, 'targets': eval_targets})
        return train_dataloader, eval_dataloader
