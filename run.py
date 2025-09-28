#!/usr/bin/env python3

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

    extracted = download_and_extract(urls, raw_dir, extract_dir, remove_zips=False)
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

        if output_dir.exists():
            shard_files = list(output_dir.glob("*.pt"))
            if shard_files:
                logger.info(f"{split_name} already processed ({len(shard_files)} shards)")
                shard_dirs[split_name] = output_dir
                continue

        stats = create_shards(
            input_dirs=[split_path],
            output_dir=output_dir,
            samples_per_shard=config['data']['samples_per_shard'],
            num_workers=config['data']['processing_workers'],
            rotations=config['data']['rotations'],
            target_size=config['data']['image_size']
        )

        verification = verify_shards(output_dir)
        logger.info(f"Verified {split_name}: {verification['total_samples']} samples")
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
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config.get('logging', {}).get('logs_dir', 'logs'), args.log_level)

    if 'all' in args.steps:
        steps_to_run = ['download', 'process', 'train', 'export']
    else:
        steps_to_run = args.steps

    start_time = time.time()
    results = {}

    # Download
    if 'download' in steps_to_run:
        splits = step_download(config)
        results['splits'] = splits
    else:
        extract_dir = Path(config['data']['extracted_dir'])
        splits = verify_dataset(extract_dir)
        results['splits'] = splits

    # Process
    if 'process' in steps_to_run:
        shard_dirs = step_process(config, results['splits'])
        results['shard_dirs'] = shard_dirs
    else:
        processed_dir = Path(config['data']['processed_dir'])
        shard_dirs = {}
        for split_name in ['train2017', 'val2017', 'test2017']:
            shard_path = processed_dir / split_name
            if shard_path.exists():
                shard_dirs[split_name] = shard_path
        results['shard_dirs'] = shard_dirs

    # Train
    if 'train' in steps_to_run:
        best_model = step_train(config, results['shard_dirs'])
        results['best_model'] = best_model

    # Export
    if 'export' in steps_to_run:
        if 'best_model' not in results:
            checkpoint_dir = Path(config['training']['output_dir'])
            best_model = checkpoint_dir / "best_model.pth"
            if not best_model.exists():
                raise FileNotFoundError("No trained model found")
            results['best_model'] = best_model

        exported = step_export(config, results['best_model'])
        results['exported'] = exported

    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline complete in {elapsed_time/60:.2f} minutes")

    if 'best_model' in results:
        logger.info(f"Model: {results['best_model']}")

    if 'exported' in results and results['exported']:
        for fmt, path in results['exported'].items():
            logger.info(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()