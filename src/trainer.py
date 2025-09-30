"""Training utilities for RotateMate."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from timm import create_model
from timm.data import resolve_model_data_config
from tqdm import tqdm
import kornia.augmentation as K

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    """Wraps model with normalization for end-to-end inference."""

    def __init__(self, model, mean, std, channels_last=False):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.channels_last = channels_last

    def forward(self, x):
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return self.model((x - self.mean) / self.std)


class Trainer:
    """Handles model training with automatic mixed precision and checkpointing."""

    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()

        # Directories
        self.checkpoint_dir = Path(config['output_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.get('logs_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self.best_metric = 0.0
        self.patience_counter = 0
        self.metrics = {'train_loss': [], 'val_acc': [], 'learning_rates': []}

        # GPU-accelerated data augmentation
        self.train_aug = nn.Sequential(
            K.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
        ).to(self.device)

        # Loss function with label smoothing
        label_smoothing = config.get('label_smoothing', 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Config shortcuts
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.channels_last = self.device.type == "cuda" and config.get('channels_last', True)
        self.use_compile = self.device.type == "cuda" and config.get('compile', False)

    def _setup_device(self):
        """Configure device and performance optimizations."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("Running on CPU - training will be slow")

        return device

    def setup_model(self, model_config):
        """Create and configure model with normalization wrapper."""
        logger.info("Loading model: %s", model_config['name'])

        # Create base model
        model_core = create_model(
            model_config['name'],
            pretrained=model_config.get('pretrained', True),
            num_classes=model_config['num_classes'],
            drop_rate=model_config.get('drop_rate', 0.0),
            drop_path_rate=model_config.get('drop_path_rate', 0.0),
        )

        # Get normalization stats
        data_cfg = resolve_model_data_config(model_core)
        mean = model_config.get('normalize', {}).get('mean', data_cfg.get('mean', (0.485, 0.456, 0.406)))
        std = model_config.get('normalize', {}).get('std', data_cfg.get('std', (0.229, 0.224, 0.225)))

        # Wrap with normalization and channels_last handling
        model = ModelWrapper(model_core, mean, std, channels_last=self.channels_last)

        # Apply optimizations
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        model = model.to(self.device)

        if self.use_compile:
            try:
                compile_mode = self.config.get('compile_mode', 'reduce-overhead')
                model = torch.compile(model, mode=compile_mode, fullgraph=True)
                logger.info("Model compiled with mode=%s, fullgraph=True", compile_mode)
            except Exception as err:
                logger.warning("torch.compile failed: %s. Falling back to eager mode.", err)

        self.model = model
        return model

    def setup_optimizer(self):
        """Create optimizer and learning rate scheduler with warmup."""
        # Use fused AdamW on CUDA for better performance
        fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            fused=fused
        )
        if fused:
            logger.info("Using fused AdamW optimizer")

        # Learning rate schedule with optional warmup
        warmup_epochs = self.config.get('warmup_epochs', 1)
        total_epochs = self.config['epochs']

        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=1e-6
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            logger.info("Using warmup (%d epochs) + cosine annealing", warmup_epochs)
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=1e-6
            )

        return self.optimizer, self.scheduler

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model accuracy on a dataloader."""
        self.model.eval()
        correct = total = 0

        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            predictions = self.model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return correct / total

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Apply augmentation on GPU
            images = self.train_aug(images)

            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                loss = self.criterion(self.model(images), labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save training checkpoint with automatic cleanup of old checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'metrics': self.metrics
        }

        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            logger.info("New best model: %.4f", val_acc)

        # Clean up old checkpoints
        keep_n = self.config.get('keep_n_checkpoints', 3)
        old_checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))[:-keep_n]
        for ckpt in old_checkpoints:
            ckpt.unlink()

        return checkpoint_path

    def train(self, train_loader, val_loader, model_config):
        """Run full training loop with early stopping."""
        self.setup_model(model_config)
        self.setup_optimizer()

        epochs = self.config['epochs']
        patience = self.config.get('patience', 3)

        logger.info("Training for %d epochs", epochs)
        logger.info("Train: %d samples | Val: %d samples", len(train_loader.dataset), len(val_loader.dataset))

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            lr = self.optimizer.param_groups[0]['lr']

            # Track metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['learning_rates'].append(lr)

            logger.info("Epoch %d/%d - Loss: %.4f, Val: %.4f, LR: %.2e", epoch, epochs, train_loss, val_acc, lr)

            # Save checkpoint and check for improvement
            is_best = val_acc > self.best_metric
            if is_best:
                self.best_metric = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_acc, is_best)

            # Early stopping
            if self.patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

            self.scheduler.step()

        # Save final metrics
        with open(self.log_dir / "training_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info("Training complete. Best validation: %.4f", self.best_metric)
        return self.checkpoint_dir / "best_model.pth"
