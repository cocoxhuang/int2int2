#!/usr/bin/env python3
"""Reproduce Table 11: per-coefficient and joint validation accuracy.

    python evaluate.py --config configs/w1w2w3.yaml --session cache/sesh_20260623_000821

Loads a trained checkpoint, rebuilds the same 80/20 split the run used (the
split is a function of the seed and the full dataset, so the whole parquet is
read), and decodes the validation set greedily.
"""
import argparse
import glob
import os

import pandas as pd
import torch

from src.data.dataset import Dataset
from src.data.tokenizer import BaseTokenizer
from src.model.transformer import Transformer
from src.training.evaluator import Evaluator
from src.utils.config import load_config


def latest_session(cache_dir):
    sessions = sorted(glob.glob(os.path.join(cache_dir, "sesh_*")))
    if not sessions:
        raise SystemExit(f"no sessions found in {cache_dir}/")
    return sessions[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None,
                    help="config to evaluate with (default: the one saved in --session)")
    ap.add_argument('--session', default=None,
                    help="training session directory (default: the most recent one)")
    ap.add_argument('--checkpoint', default='best_model.pth')
    ap.add_argument('--eval-size', type=int, default=None,
                    help="cap the validation set (default: the config's eval_size, "
                         "which is what Table 11 used; 0 evaluates the whole 20% split)")
    ap.add_argument('--batch-size', type=int, default=None,
                    help="decoding batch size; affects throughput only, not results")
    args = ap.parse_args()

    session = args.session or latest_session('cache')
    config = load_config(args.config or os.path.join(session, 'config'))
    print(f"session    : {session}")
    print(f"checkpoint : {args.checkpoint}")

    base = config['data']['base']
    tokenizer = BaseTokenizer(base=base)
    data = pd.read_parquet(config['data']['data_path'])
    print(f"data       : {config['data']['data_path']} ({len(data):,} isogeny classes)")

    eval_size = config['training']['eval_size'] if args.eval_size is None else args.eval_size
    if eval_size == 0:          # no cap: the entire 20% validation split
        eval_size = len(data)
    batch_size = args.batch_size or config['training']['batch_size']
    dataset = Dataset(data, tokenizer=tokenizer, base=base,
                      seed=config['data']['seed'],
                      batch_size=batch_size, eval_size=eval_size)

    model = Transformer(
        src_vocab_size=config['model']['src_vocab_size'],
        tgt_vocab_size=config['model']['tgt_vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        max_len=config['model']['max_len'],
        dropout=config['model']['dropout'],
        architecture=config['model']['architecture'],
        is_sinusoidal=config['model']['is_sinusoidal'])

    device = torch.device('cuda' if torch.cuda.is_available() and
                          config['device']['use_cuda'] else 'cpu')
    path = os.path.join(session, args.checkpoint)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

    evaluator = Evaluator(model, dataset.eval_dataloader,
                          torch.nn.CrossEntropyLoss(), device, tokenizer=tokenizer)
    max_new_tokens = dataset.data['targets'].size(1) - 1
    stats = evaluator.evaluate_per_coefficient(max_new_tokens)

    k = len(stats['accuracy'])
    names = [f"w{i}" for i in range(1, k + 1)]
    print(f"\nValidation set: {stats['n']:,} isogeny classes, greedy decoding\n")
    head = "".join(f"{n:>10}" for n in names) + f"{'joint':>10}"
    print(f"{'':14}{head}")
    print(f"{'transformer':14}" + "".join(f"{a * 100:9.1f}%" for a in stats['accuracy'])
          + f"{stats['joint'] * 100:9.1f}%")
    print(f"{'baseline':14}" + "".join(f"{b * 100:9.1f}%" for b in stats['baseline']))


if __name__ == "__main__":
    main()
