import argparse
from src.model.transformer import Transformer
from src.data.dataset import Dataset
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.data.tokenizer import BaseTokenizer
from src.utils.logger import Logger
import os
import pandas as pd

def main(config_path, resume_from=None):
    config = load_config(config_path)
    
    if resume_from:
        # Load configuration from the resume session
        resume_config_path = os.path.join(resume_from, "config")
        if os.path.exists(resume_config_path):
            config = load_config(resume_config_path)
            # Use the old configuration but allow some overrides from new config
            logger = Logger(resume_from=resume_from)
            logger.info(f"Resuming training from session: {resume_from}")
        else:
            logger = Logger(cache_dir=config['training']['cache_dir'], name=config['training'].get('experiment_name', 'experiment'))
            logger.error(f"Resume session config not found: {resume_config_path}")
            return
    else:
        # Create new logger for fresh training
        cache_dir = config['training']['cache_dir']
        logger = Logger(cache_dir=cache_dir)
    
    # get the csv/parquet file from config if it exists
    data_path = config['data'].get('data_path', None)  # None will use default path with n
    if data_path:
        logger.info(f"Using data path from config: {data_path}")
    else:
        raise ValueError("No data path specified in config.")
    if data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    else:
        raise ValueError("Data file must be a .parquet or .csv file")

    # now create a torch dataset and dataloader
    base = config['data'].get('base', 1000)
    tokenizer = BaseTokenizer(base=base)
    dataset = Dataset(data, tokenizer=tokenizer, base=base, batch_size=config['training']['batch_size'], eval_size=config['training']['eval_size'])

    # Initialize the model
    model = Transformer(
        src_vocab_size=config['model']['src_vocab_size'],
        tgt_vocab_size=config['model']['tgt_vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        max_len=config['model']['max_len'],  # Use actual data length
        dropout=config['model']['dropout'],
        architecture=config['model']['architecture'],
        is_sinusoidal=config['model']['is_sinusoidal']
    )

    # Initialize the trainer (it will create its own logger)
    trainer = Trainer(model, dataset, config, logger, resume_from=resume_from)

    # Start training
    results = trainer.train()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to training session to resume from (e.g., cache/sesh_20250804_052923)')
    # parser.add_argument('--list-sessions', action='store_true', help='List all available training sessions')
    args = parser.parse_args()

    main(args.config, resume_from=args.resume)