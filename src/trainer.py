"""Simple trainer for rotation classification."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm import create_model
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Simple trainer with standard practices."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_dir = Path(config['output_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = 0.0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

    def setup_model(self, model_config):
        """Create model."""
        model = create_model(
            model_config['name'],
            pretrained=model_config.get('pretrained', True),
            num_classes=model_config['num_classes']
        )
        model = model.to(self.device)

        logger.info("Created model: %s", model_config['name'])
        return model

    def setup_optimizer(self):
        """Create optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )

        logger.info("Optimizer: AdamW, LR: %.2e", self.config['learning_rate'])

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate accuracy."""
        self.model.eval()
        correct = total = 0

        for images, labels in tqdm(dataloader, desc="Eval", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return correct / total

    def train_epoch(self, dataloader):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(dataloader, desc="Train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, model_config):
        """Main training loop."""
        self.model = self.setup_model(model_config)
        self.setup_optimizer()

        patience = self.config.get('patience', 3)
        no_improve = 0

        for epoch in range(self.config['epochs']):
            logger.info("Epoch %d/%d", epoch + 1, self.config['epochs'])

            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            self.scheduler.step()

            logger.info("Loss: %.4f | Val Acc: %.4f", train_loss, val_acc)

            # Save best model
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(self.model.state_dict(), best_path)
                logger.info("Saved best model: %.4f", val_acc)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping")
                    break

        return self.checkpoint_dir / "best_model.pth", self.best_metric