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


def step_train(config, splits):
    logger = logging.getLogger(__name__)
    logger.info("Step 2: Training model")

    data_cfg = config['data']
    train_dir = splits.get(data_cfg.get('train_split', 'train2017'))
    val_dir = splits.get(data_cfg.get('val_split', 'val2017'))
    test_dir = splits.get(data_cfg.get('test_split', 'test2017'))

    if not train_dir or not val_dir:
        raise ValueError("Missing train/val datasets")

    rotations = data_cfg.get('rotations', [0, 90, 180, 270])
    image_size = data_cfg.get('image_size', 256)
    dataloaders = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        rotations=rotations,
        image_size=image_size,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        prefetch_factor=config['training'].get('prefetch_factor', 2)
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
    logger.info("Step 3: Exporting model")

    output_dir = Path(config['training']['output_dir']) / "exports"
    exported = export_model(checkpoint_path, config['export'], config['model'], output_dir)
    logger.info(f"Exported: {list(exported.keys())}")
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--steps', nargs='+',
                       choices=['download', 'train', 'export', 'all'],
                       default=['all'])
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config['logging']['logs_dir'])

    if 'all' in args.steps:
        steps_to_run = ['download', 'train', 'export']
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

        # Train
        if 'train' in steps_to_run:
            results['best_model'] = step_train(config, results['splits'])

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
