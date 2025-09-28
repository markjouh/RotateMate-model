#!/usr/bin/env python3
"""
Main orchestrator for training pipeline.
"""

import sys
import argparse
import logging
import yaml
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from downloader import download_and_extract, verify_dataset
from processor import create_shards, verify_shards
from dataset import create_dataloaders
from trainer import Trainer, export_model


def setup_logging(log_dir, level="INFO"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def step_download(config):
    logger = logging.getLogger(__name__)
    logger.info("Step 1: Downloading datasets")

    urls = config['data']['urls']
    raw_dir = Path(config['data']['raw_dir'])
    extract_dir = Path(config['data']['extracted_dir'])

    download_and_extract(urls, raw_dir, extract_dir)
    splits = verify_dataset(extract_dir)
    logger.info(f"Found {len(splits)} dataset splits")
    return splits


def step_process(config, splits):
    logger = logging.getLogger(__name__)
    logger.info("Step 2: Processing images")

    processed_dir = Path(config['data']['processed_dir'])
    shard_dirs = {}

    for split_name, split_path in splits.items():
        logger.info(f"Processing {split_name}...")
        output_dir = processed_dir / split_name

        if output_dir.exists() and list(output_dir.glob("*.pt")):
            logger.info(f"{split_name} already processed")
            shard_dirs[split_name] = output_dir
            continue

        create_shards(
            input_dirs=[split_path],
            output_dir=output_dir,
            samples_per_shard=config['data']['samples_per_shard'],
            num_workers=config['data']['processing_workers'],
            rotations=config['data']['rotations'],
            target_size=config['data']['image_size']
        )

        verify_shards(output_dir)
        shard_dirs[split_name] = output_dir

    return shard_dirs


def step_train(config, shard_dirs):
    logger = logging.getLogger(__name__)
    logger.info("Step 3: Training model")

    train_dir = shard_dirs.get('train2017')
    val_dir = shard_dirs.get('val2017')
    test_dir = shard_dirs.get('test2017')

    if not train_dir or not val_dir:
        raise ValueError("Missing train/val datasets")

    dataloaders = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    trainer = Trainer(config['training'])
    best_model_path = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        model_config=config['model']
    )

    if 'test' in dataloaders:
        test_acc = trainer.evaluate(dataloaders['test'])
        logger.info(f"Test accuracy: {test_acc:.4f}")

    return best_model_path


def step_export(config, checkpoint_path):
    logger = logging.getLogger(__name__)
    logger.info("Step 4: Exporting model")

    output_dir = Path(config['training']['output_dir']) / "exports"
    exported = export_model(checkpoint_path, config['export'], output_dir)
    logger.info(f"Exported: {list(exported.keys())}")
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--steps', nargs='+',
                       choices=['download', 'process', 'train', 'export', 'all'],
                       default=['all'])
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config['logging']['logs_dir'])

    if 'all' in args.steps:
        steps_to_run = ['download', 'process', 'train', 'export']
    else:
        steps_to_run = args.steps

    start_time = time.time()
    results = {}

    try:
        # Download
        if 'download' in steps_to_run:
            results['splits'] = step_download(config)
        else:
            results['splits'] = verify_dataset(Path(config['data']['extracted_dir']))

        # Process
        if 'process' in steps_to_run:
            results['shard_dirs'] = step_process(config, results['splits'])
        else:
            processed_dir = Path(config['data']['processed_dir'])
            shard_dirs = {}
            for split in ['train2017', 'val2017', 'test2017']:
                path = processed_dir / split
                if path.exists():
                    shard_dirs[split] = path
            results['shard_dirs'] = shard_dirs

        # Train
        if 'train' in steps_to_run:
            results['best_model'] = step_train(config, results['shard_dirs'])

        # Export
        if 'export' in steps_to_run:
            if 'best_model' not in results:
                best_model = Path(config['training']['output_dir']) / "best_model.pth"
                if not best_model.exists():
                    raise FileNotFoundError("No model to export")
                results['best_model'] = best_model
            results['exported'] = step_export(config, results['best_model'])

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    elapsed = time.time() - start_time
    logger.info(f"Complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()