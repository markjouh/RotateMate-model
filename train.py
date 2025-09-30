#!/usr/bin/env python3
"""Train RotateMate model with optional hyperparameter sweep."""

import argparse
import copy
import json
import logging
import yaml
from pathlib import Path
from datetime import datetime
from itertools import product

from src import download_and_extract, verify_dataset, create_dataloaders, Trainer, export_model
from src.preprocess import preprocess_split_gpu


def setup_logging(log_dir, log_to_file=True):
    """Setup logging to console and optionally to file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"training_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def run_training(config, output_dir, dataloaders, seed=None):
    """Run a single training job."""
    import torch
    import numpy as np
    import random

    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Train
    trainer = Trainer(config['training'])
    best_model_path = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        model_config=config['model']
    )

    return best_model_path, trainer.best_metric


def main():
    parser = argparse.ArgumentParser(description="Train RotateMate model")
    parser.add_argument('--config', default='configs/h100.yaml', help='Config file path')
    parser.add_argument('--steps', nargs='+', choices=['download', 'train', 'export', 'all'],
                       default=['all'], help='Pipeline steps to run')

    # Hyperparameter sweep options
    parser.add_argument('--learning-rate', nargs='+', type=float, help='Learning rates to sweep')
    parser.add_argument('--weight-decay', nargs='+', type=float, help='Weight decays to sweep')
    parser.add_argument('--seeds', nargs='+', type=int, help='Random seeds for sweep')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine if this is a sweep
    is_sweep = any([args.learning_rate, args.weight_decay, args.seeds])

    if is_sweep:
        # Sweep mode
        learning_rates = args.learning_rate or [config['training']['learning_rate']]
        weight_decays = args.weight_decay or [config['training'].get('weight_decay', 1e-4)]
        seeds = args.seeds or [0]

        # Create sweep directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = Path(config['training']['output_dir']) / f"sweep_{timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logging(sweep_dir, log_to_file=False)
        logger.info("Starting hyperparameter sweep")
        logger.info(f"Learning rates: {learning_rates}")
        logger.info(f"Weight decays: {weight_decays}")
        logger.info(f"Seeds: {seeds}")

        # Download data if needed
        data_cfg = config['data']
        if 'download' in args.steps or 'all' in args.steps:
            logger.info("Downloading datasets")
            download_and_extract(data_cfg['urls'], data_cfg['raw_dir'], data_cfg['extracted_dir'])

        # Preprocess if needed
        rotations = [0, 90, 180, 270]
        image_size = data_cfg.get('image_size', 256)
        extracted_dir = Path(data_cfg['extracted_dir'])

        preprocess_cfg = data_cfg.get('preprocessing', {})
        preprocess_batch_size = preprocess_cfg.get('batch_size', 512)
        preprocess_workers = preprocess_cfg.get('num_workers', 8)

        for split_name in ['train2017', 'val2017']:
            split_path = extracted_dir / split_name
            preprocessed_dir = extracted_dir / f"{split_name}_preprocessed"

            if split_path.exists() and not (preprocessed_dir.exists() and any(preprocessed_dir.glob("*.pt"))):
                logger.info(f"Preprocessing {split_name}")
                preprocess_split_gpu(split_path, preprocessed_dir, image_size=image_size,
                                   batch_size=preprocess_batch_size, num_workers=preprocess_workers)

        # Create dataloaders once
        splits = verify_dataset(data_cfg['extracted_dir'])
        train_dir = splits.get(data_cfg.get('train_split', 'train2017'))
        val_dir = splits.get(data_cfg.get('val_split', 'val2017'))

        if not train_dir or not val_dir:
            raise ValueError("Missing train/val datasets")

        logger.info("Creating dataloaders (reused across all experiments)")
        dataloaders = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            rotations=rotations,
            image_size=image_size,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'],
            prefetch_factor=config['training']['prefetch_factor']
        )

        # Run experiments
        results = []
        combinations = list(product(learning_rates, weight_decays, seeds))

        for idx, (lr, wd, seed) in enumerate(combinations, 1):
            exp_name = f"lr{lr:.0e}_wd{wd:.0e}_seed{seed}"
            exp_dir = sweep_dir / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {idx}/{len(combinations)}: {exp_name}")
            logger.info(f"{'='*60}")

            # Configure experiment
            exp_config = copy.deepcopy(config)
            exp_config['training']['learning_rate'] = lr
            exp_config['training']['weight_decay'] = wd
            exp_config['training']['output_dir'] = str(exp_dir)
            exp_config['training']['logs_dir'] = str(exp_dir / "logs")

            try:
                best_model_path, best_val_acc = run_training(exp_config, exp_dir, dataloaders, seed)
                results.append({
                    'exp_name': exp_name,
                    'lr': lr,
                    'wd': wd,
                    'seed': seed,
                    'best_val_acc': best_val_acc,
                    'best_model_path': str(best_model_path),
                })
                logger.info(f"Completed: {exp_name} - Val Acc: {best_val_acc:.4f}")
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                results.append({
                    'exp_name': exp_name,
                    'lr': lr,
                    'wd': wd,
                    'seed': seed,
                    'best_val_acc': 0.0,
                    'error': str(e)
                })

        # Save and report results
        results_sorted = sorted(results, key=lambda x: x.get('best_val_acc', 0), reverse=True)

        summary_path = sweep_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'sweep_config': {
                    'learning_rates': learning_rates,
                    'weight_decays': weight_decays,
                    'seeds': seeds,
                },
                'results': results_sorted,
                'best': results_sorted[0] if results_sorted else None
            }, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("Sweep complete!")
        logger.info(f"Results saved to: {summary_path}")

        if results_sorted:
            best = results_sorted[0]
            logger.info(f"\nBest result:")
            logger.info(f"  {best['exp_name']}: {best['best_val_acc']:.4f}")
            logger.info(f"  LR: {best['lr']:.2e}, WD: {best['wd']:.2e}, Seed: {best['seed']}")

            # Symlink best model
            if 'best_model_path' in best and Path(best['best_model_path']).exists():
                import os
                standard_best = Path(config['training']['output_dir']) / "best_model.pth"
                if standard_best.exists() or standard_best.is_symlink():
                    standard_best.unlink()
                os.symlink(Path(best['best_model_path']).absolute(), standard_best)
                logger.info(f"  Best model: {standard_best}")

            logger.info(f"\nTop 3 results:")
            for i, result in enumerate(results_sorted[:3], 1):
                logger.info(f"  {i}. {result['exp_name']}: {result['best_val_acc']:.4f}")

    else:
        # Single training mode
        steps = ['download', 'train', 'export'] if 'all' in args.steps else args.steps
        data_cfg = config['data']
        logger = setup_logging(config['logging']['logs_dir'])

        # Step 1: Download
        if 'download' in steps:
            logger.info("Step 1: Downloading datasets")
            download_and_extract(data_cfg['urls'], data_cfg['raw_dir'], data_cfg['extracted_dir'])

        # Preprocess if needed
        rotations = [0, 90, 180, 270]
        image_size = data_cfg.get('image_size', 256)
        extracted_dir = Path(data_cfg['extracted_dir'])

        preprocess_cfg = data_cfg.get('preprocessing', {})
        preprocess_batch_size = preprocess_cfg.get('batch_size', 512)
        preprocess_workers = preprocess_cfg.get('num_workers', 8)

        for split_name in ['train2017', 'val2017']:
            split_path = extracted_dir / split_name
            preprocessed_dir = extracted_dir / f"{split_name}_preprocessed"

            if split_path.exists() and not (preprocessed_dir.exists() and any(preprocessed_dir.glob("*.pt"))):
                logger.info(f"Preprocessing {split_name}")
                preprocess_split_gpu(split_path, preprocessed_dir, image_size=image_size,
                                   batch_size=preprocess_batch_size, num_workers=preprocess_workers)

        # Step 2: Train
        best_model_path = None
        if 'train' in steps:
            logger.info("Step 2: Training model")
            splits = verify_dataset(data_cfg['extracted_dir'])

            train_dir = splits.get(data_cfg.get('train_split', 'train2017'))
            val_dir = splits.get(data_cfg.get('val_split', 'val2017'))

            if not train_dir or not val_dir:
                raise ValueError("Missing train/val datasets")

            dataloaders = create_dataloaders(
                train_dir=train_dir,
                val_dir=val_dir,
                rotations=rotations,
                image_size=image_size,
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                pin_memory=config['training']['pin_memory'],
                prefetch_factor=config['training']['prefetch_factor']
            )

            best_model_path, _ = run_training(config, config['training']['output_dir'], dataloaders)

        # Step 3: Export
        if 'export' in steps:
            logger.info("Step 3: Exporting model")

            if not best_model_path:
                best_model_path = Path(config['training']['output_dir']) / "best_model.pth"
                if not best_model_path.exists():
                    raise FileNotFoundError(f"No model found at {best_model_path}")

            output_dir = Path(config['training']['output_dir']) / "exports"
            exported = export_model(
                best_model_path,
                config['export'],
                config['model'],
                output_dir,
                image_size=data_cfg.get('image_size', 256)
            )
            logger.info("Exported: %s", list(exported.keys()))

        logger.info("Complete!")


if __name__ == "__main__":
    main()