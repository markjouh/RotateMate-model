"""
Trains MobileViT V2 model on rotated images using mixed precision and exports to CoreML/ONNX.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    def __init__(self, model_core, mean, std):
        super().__init__()
        self.model = model_core
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(pixel_values=x).logits


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # H100: Enable TF32 for Hopper architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.checkpoint_dir = Path(config['output_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.get('logs_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_metric = 0.0
        self.patience_counter = 0
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        self.metrics = {
            'train_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    def setup_model(self, model_config):
        from transformers import MobileViTV2ForImageClassification
        warnings.filterwarnings("ignore", message=".*slow image processor.*")

        model_id = model_config['name']
        num_classes = model_config['num_classes']

        logger.info(f"Loading {model_id}")

        model_core = MobileViTV2ForImageClassification.from_pretrained(
            model_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            label2id={str(i): i for i in range(num_classes)},
            id2label={i: str(i) for i in range(num_classes)},
        )

        mean = model_config['normalize']['mean']
        std = model_config['normalize']['std']
        self.model = ModelWrapper(model_core, mean, std).to(self.device)

        return self.model

    def setup_optimizer(self):
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-4)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        epochs = self.config['epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

        self.scaler = GradScaler("cuda")

        return self.optimizer, self.scheduler

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return correct / max(1, total)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        criterion = nn.CrossEntropyLoss()
        accumulation_steps = self.gradient_accumulation_steps

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(images)
                loss = criterion(logits, labels) / accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'metrics': self.metrics
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {val_acc:.4f}")

        keep_n = self.config.get('keep_n_checkpoints', 3)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_n:
            for old_ckpt in checkpoints[:-keep_n]:
                old_ckpt.unlink()

        return checkpoint_path

    def train(self, train_loader, val_loader, model_config):
        self.setup_model(model_config)
        self.setup_optimizer()

        epochs = self.config['epochs']
        patience = self.config.get('patience', 3)

        logger.info(f"Training for {epochs} epochs")
        logger.info(f"Train: {len(train_loader.dataset)} samples")
        logger.info(f"Val: {len(val_loader.dataset)} samples")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['learning_rates'].append(current_lr)

            logger.info(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}, Val: {val_acc:.4f}, LR: {current_lr:.2e}")

            is_best = val_acc > self.best_metric
            if is_best:
                self.best_metric = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_acc, is_best)

            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            self.scheduler.step()

        metrics_path = self.log_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Best validation: {self.best_metric:.4f}")
        return self.checkpoint_dir / "best_model.pth"


def export_model(checkpoint_path, export_config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']

    model_config = {
        'name': 'apple/mobilevitv2-1.0-imagenet1k-256',
        'num_classes': 4,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    from transformers import MobileViTV2ForImageClassification
    model_core = MobileViTV2ForImageClassification.from_pretrained(
        model_config['name'],
        num_labels=model_config['num_classes'],
        ignore_mismatched_sizes=True
    )
    model = ModelWrapper(model_core,
                        model_config['normalize']['mean'],
                        model_config['normalize']['std'])
    model.load_state_dict(model_state)
    model.eval()

    if export_config.get('coreml', {}).get('enabled', False):
        try:
            import coremltools as ct

            logger.info("Exporting Core ML...")
            example_input = torch.randn(1, 3, 256, 256)
            traced = torch.jit.trace(model, example_input)

            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                classifier_config=ct.ClassifierConfig(
                    class_labels=[str(i) for i in range(model_config['num_classes'])]
                ),
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.iOS17
            )

            if export_config['coreml'].get('fp16', True):
                mlmodel = ct.models.utils.convert_neural_network_weights_to_fp16(mlmodel)

            mlmodel_path = output_dir / "model.mlmodel"
            mlmodel.save(str(mlmodel_path))
            exported['coreml'] = mlmodel_path
            logger.info(f"Exported: {mlmodel_path}")

        except ImportError:
            logger.warning("coremltools not installed")
        except Exception as e:
            logger.error(f"Core ML export failed: {e}")

    if export_config.get('onnx', {}).get('enabled', False):
        try:
            logger.info("Exporting ONNX...")
            example_input = torch.randn(1, 3, 256, 256)
            onnx_path = output_dir / "model.onnx"

            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                opset_version=export_config['onnx'].get('opset_version', 17),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
            )

            exported['onnx'] = onnx_path
            logger.info(f"Exported: {onnx_path}")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

    metadata_path = output_dir / "export_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'val_acc': float(checkpoint['val_acc']),
            'exported_formats': list(exported.keys()),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    return exported