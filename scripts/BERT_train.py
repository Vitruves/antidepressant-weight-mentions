#!/usr/bin/env python3
"""Fine-tune and evaluate a BERT-family encoder for multi-class text classification."""

import argparse
import json
import logging
import random
import signal
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, StepLR
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_scheduler
)
from tqdm import tqdm

try:
    from argparse_color_formatter import ColorHelpFormatter
except ImportError:
    ColorHelpFormatter = argparse.HelpFormatter

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s : %(message)s',
    datefmt='%H:%M'
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Received interrupt signal, shutting down gracefully...")
        self.kill_now = True


killer = GracefulKiller()


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017)."""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_prob = nn.functional.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * \
            self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def get_loss_function(args, class_weights=None, device=None):
    """Get the appropriate loss function based on arguments."""
    if args.focal_loss:
        logger.info(
            f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.label_smoothing > 0:
        logger.info(
            f"Using Label Smoothing Cross Entropy (smoothing={args.label_smoothing})")
        return LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif args.use_class_weights and class_weights is not None:
        logger.info("Using weighted Cross Entropy")
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Return None to use the model's built-in loss (original behavior)
        return None


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create a weighted random sampler for imbalanced datasets."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


class ThresholdMoving:
    """Threshold moving for imbalanced classification."""

    def __init__(self, labels: List[int]):
        self.class_counts = np.bincount(labels)
        self.total_samples = len(labels)
        self.class_priors = self.class_counts / self.total_samples
        self.optimal_thresholds = None

    def fit(self, y_true: np.ndarray, y_probs: np.ndarray):
        """Find optimal thresholds using validation data."""
        n_classes = len(self.class_counts)
        self.optimal_thresholds = np.ones(n_classes) / n_classes

        # Simple threshold adjustment based on class priors
        for i in range(n_classes):
            if self.class_priors[i] < 1.0 / n_classes:  # Minority class
                self.optimal_thresholds[i] *= 0.8  # Lower threshold
            else:  # Majority class
                self.optimal_thresholds[i] *= 1.2  # Higher threshold

        self.optimal_thresholds = self.optimal_thresholds / \
            np.sum(self.optimal_thresholds)
        logger.info(f"Optimal thresholds: {self.optimal_thresholds}")

    def predict(self, y_probs: np.ndarray) -> np.ndarray:
        """Make predictions using optimal thresholds."""
        if self.optimal_thresholds is None:
            raise ValueError("Must call fit() before predict()")

        adjusted_probs = y_probs / self.optimal_thresholds
        return np.argmax(adjusted_probs, axis=1)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataframe(input_path: Path, **kwargs) -> pd.DataFrame:
    """Load dataframe from various file formats."""
    suffix = input_path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(input_path, **kwargs)
    elif suffix == '.json':
        return pd.read_json(input_path, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(input_path, **kwargs)
        except ImportError as e:
            if 'openpyxl' in str(e):
                raise ImportError(
                    "Excel support requires openpyxl: pip install openpyxl") from e
            elif 'xlrd' in str(e):
                raise ImportError(
                    "Excel .xls support requires xlrd: pip install xlrd") from e
            raise
    elif suffix == '.parquet':
        try:
            return pd.read_parquet(input_path, **kwargs)
        except ImportError as e:
            if 'pyarrow' in str(e) or 'fastparquet' in str(e):
                raise ImportError(
                    "Parquet support requires pyarrow or fastparquet: pip install pyarrow") from e
            raise
    elif suffix == '.tsv':
        return pd.read_csv(input_path, sep='\t', **kwargs)
    elif suffix == '.jsonl':
        lines = []
        with open(input_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line.strip()))
        return pd.DataFrame(lines)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported: .csv, .json, .xlsx, .xls, .parquet, .tsv, .jsonl")


class TextDataset(Dataset):
    """Tokenised text/label pairs."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = [str(text) for text in texts]
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if pd.isna(text) or not text.strip():
            text = "[EMPTY]"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SequenceClassifier(nn.Module):
    """Sequence classifier around any Hugging Face encoder checkpoint."""

    def __init__(self, model_name_or_path: str, num_labels: int, dropout: float = 0.1,
                 trust_remote_code: bool = False):
        super().__init__()

        self.num_labels = num_labels

        try:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                trust_remote_code=trust_remote_code
            )
            self.is_hf_model = True
            logger.info(f"Loaded HuggingFace model from {model_name_or_path}")
        except Exception as e:
            logger.warning(f"Failed to load as HuggingFace model: {e}")
            try:
                from transformers import AutoModel
                self.bert_base = AutoModel.from_pretrained(
                    model_name_or_path, trust_remote_code=trust_remote_code)
                hidden_size = self.bert_base.config.hidden_size
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_size, num_labels)
                self.is_hf_model = False
                logger.info(
                    f"Loaded base model from {model_name_or_path} with custom classifier")
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise e2

    def forward(self, input_ids, attention_mask, labels=None):
        if self.is_hf_model:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = self.bert_base(
                input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            return {'loss': loss, 'logits': logits}


def resample_data(texts: List[str], labels: List[int], strategy: str = 'undersample',
                  random_state: int = 42) -> Tuple[List[str], List[int]]:
    """Balance classes by over- or undersampling."""
    if strategy == 'none':
        return texts, labels

    df = pd.DataFrame({'text': texts, 'label': labels})
    label_counts = df['label'].value_counts()

    if strategy == 'undersample':
        min_count = label_counts.min()
        balanced_dfs = []
        for label in label_counts.index:
            label_df = df[df['label'] == label]
            if len(label_df) > min_count:
                label_df = label_df.sample(
                    n=min_count, random_state=random_state)
            balanced_dfs.append(label_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    elif strategy == 'oversample':
        max_count = label_counts.max()
        balanced_dfs = []
        for label in label_counts.index:
            label_df = df[df['label'] == label]
            if len(label_df) < max_count:
                additional_samples = resample(
                    label_df,
                    n_samples=max_count - len(label_df),
                    random_state=random_state,
                    replace=True
                )
                label_df = pd.concat(
                    [label_df, additional_samples], ignore_index=True)
            balanced_dfs.append(label_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    balanced_df = balanced_df.sample(
        frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Original distribution: {dict(label_counts)}")
    logger.info(
        f"Resampled ({strategy}) distribution: {dict(balanced_df['label'].value_counts())}")

    return balanced_df['text'].tolist(), balanced_df['label'].tolist()


def get_scheduler_fn(scheduler_name: str, optimizer, num_training_steps: int, **kwargs):
    """Get learning rate scheduler."""
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'linear':
        return get_scheduler('linear', optimizer,
                             num_warmup_steps=kwargs.get('warmup_steps', 0),
                             num_training_steps=num_training_steps)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_training_steps)
    elif scheduler_name == 'exponential':
        return ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.95))
    elif scheduler_name == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 10),
                      gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', patience=kwargs.get('patience', 5),
                                 factor=kwargs.get('factor', 0.5))
    elif scheduler_name == 'none':
        return None
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}, using linear")
        return get_scheduler('linear', optimizer,
                             num_warmup_steps=kwargs.get('warmup_steps', 0),
                             num_training_steps=num_training_steps)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions,
                           average='weighted', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    kappa = cohen_kappa_score(labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'cohen_kappa': kappa
    }

    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    for i, f1 in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = f1

    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device,
                use_fp16: bool = False, loss_function=None):
    """Train for one epoch with optional class weighting."""
    model.train()
    total_loss = 0
    num_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    progress_bar = tqdm(dataloader, desc="Training", leave=False,
                        position=0, colour='blue')

    for batch in progress_bar:
        if killer.kill_now:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        if use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)

                if loss_function is not None:
                    loss = loss_function(outputs['logits'], batch['labels'])
                else:
                    loss = outputs['loss']
        else:
            outputs = model(**batch)

            if loss_function is not None:
                loss = loss_function(outputs['logits'], batch['labels'])
            else:
                loss = outputs['loss']

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, dataloader, device, use_fp16: bool = False,
             threshold_mover=None, return_probabilities: bool = False):
    """Evaluate the model and return predictions, labels, and loss."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False,
                        position=0, colour='green')

    with torch.no_grad():
        for batch in progress_bar:
            if killer.kill_now:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            logits = outputs['logits']
            if outputs['loss'] is not None:
                loss = outputs['loss']
                total_loss += loss.item()

            probabilities = torch.softmax(logits, dim=-1)
            all_probabilities.extend(probabilities.cpu().numpy())

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            num_batches += 1

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    if threshold_mover is not None:
        if threshold_mover.optimal_thresholds is None:
            threshold_mover.fit(all_labels, all_probabilities)

        all_predictions = threshold_mover.predict(all_probabilities)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    if return_probabilities:
        return all_predictions, all_labels, avg_loss, all_probabilities
    else:
        return all_predictions, all_labels, avg_loss


class BERTPredictor:
    """Prediction class for trained BERT models."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)

        self.label_map = self.config['label_mapping']
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        num_labels = len(self.label_map)
        model_config = self.config.get('model_config', {})

        self.model = SequenceClassifier(
            model_name_or_path=self.config.get(
                'bert_model', 'bert-base-uncased'),
            num_labels=num_labels,
            dropout=model_config.get('dropout', 0.1)
        )

        state_dict = torch.load(self.model_dir / 'best_model.pt',
                                map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str], return_probabilities: bool = False) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        """Make predictions on texts."""
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = TextDataset(texts, dummy_labels, self.tokenizer,
                              self.config.get('max_length', 512))
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        predictions, _, _ = evaluate(self.model, dataloader, self.device,
                                     self.config.get('fp16', False))

        predicted_labels = [self.reverse_label_map[pred]
                            for pred in predictions]

        if return_probabilities:
            all_probs = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    probs = torch.softmax(outputs['logits'], dim=-1)
                    all_probs.extend(probs.cpu().numpy())

            return predicted_labels, np.array(all_probs)

        return predicted_labels


def save_complete_model(model, tokenizer, args, results, label_mapping, output_dir: Path):
    """Save complete model with all necessary components for prediction."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / 'best_model.pt')

    tokenizer.save_pretrained(output_dir)

    config = {
        **results,
        'bert_model': args.model_path if args.model_path else args.hf_model_name,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'fp16': args.fp16,
        'label_mapping': label_mapping,
        'model_config': {
            'dropout': args.dropout,
            'num_labels': len(label_mapping)
        },
        'training_args': vars(args)
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    prediction_script = '''#!/usr/bin/env python3
"""
Prediction script for BERT model.
Usage: python predict.py --model-dir path/to/model --input "Your text here"
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from scripts
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.training.BERT_train import BERTPredictor

def main():
    parser = argparse.ArgumentParser(description="Make predictions with BERT model")
    parser.add_argument('--model-dir', type=Path, required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input text to classify')
    parser.add_argument('--probabilities', action='store_true',
                       help='Return prediction probabilities')
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = BERTPredictor(args.model_dir)
    
    # Make prediction
    if args.probabilities:
        labels, probs = predictor.predict([args.input], return_probabilities=True)
        print(f"Predicted label: {labels[0]}")
        print("Probabilities:")
        for i, (orig_label, prob) in enumerate(zip(predictor.reverse_label_map.values(), probs[0])):
            print(f"  {orig_label}: {prob:.4f}")
    else:
        labels = predictor.predict([args.input])
        print(f"Predicted label: {labels[0]}")

if __name__ == "__main__":
    main()
'''

    with open(output_dir / 'predict.py', 'w') as f:
        f.write(prediction_script)

    (output_dir / 'predict.py').chmod(0o755)

    readme_content = f'''# BERT Text Classification Model

This directory contains a trained BERT model for text classification.

## Model Information
- **BERT Model**: {args.model_path if args.model_path else args.hf_model_name}
- **Classes**: {len(label_mapping)}
- **Test Accuracy**: {results.get('test_accuracy', 'N/A')}
- **Test F1 Macro**: {results.get('test_f1_macro', 'N/A')}

## Label Mapping
{json.dumps(label_mapping, indent=2)}

## Usage

### Command Line Prediction
```bash
python predict.py --model-dir . --input "Your text here"
```

### With Probabilities
```bash
python predict.py --model-dir . --input "Your text here" --probabilities
```

### Python API
```python
from BERT_train import BERTPredictor

# Load model
predictor = BERTPredictor('path/to/model/directory')

# Single prediction
labels = predictor.predict(["Your text here"])

# With probabilities
labels, probs = predictor.predict(["Your text here"], return_probabilities=True)
```

## Files
- `best_model.pt`: Trained model weights
- `config.json`: Complete model configuration
- `tokenizer.json`, `vocab.txt`, etc.: Tokenizer files
- `predict.py`: Standalone prediction script
- `results.json`: Training results and metrics

## Requirements
- torch
- transformers
- scikit-learn
- numpy
- pandas
'''

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    logger.info(f"Complete model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced BERT text classification with proper label handling. "
        "Designed to work consistently with graph_BERT training pipeline.",
        formatter_class=ColorHelpFormatter
    )

    input_group = parser.add_argument_group('Input/Output')
    input_mode = input_group.add_mutually_exclusive_group(required=True)
    input_mode.add_argument('-i', '--input', type=Path,
                            help='Single input data file (CSV, TSV, JSON, JSONL, Parquet, Excel)')
    input_mode.add_argument('--train-file', type=Path,
                            help='Training data file (use with --val-file)')

    input_group.add_argument('--val-file', type=Path,
                             help='Validation data file (required when using --train-file)')
    input_group.add_argument('--test-file', type=Path,
                             help='Test data file (optional)')
    input_group.add_argument('-o', '--output', type=Path, required=True,
                             help='Output directory for model and results')
    input_group.add_argument('--text-col', default='text',
                             help='Text column name (default: text)')
    input_group.add_argument('--label-col', default='label',
                             help='Label column name (default: label)')
    input_group.add_argument('--val-split', type=float, default=0.2,
                             help='Validation split ratio for single input file (default: 0.2)')
    input_group.add_argument('--test-split', type=float, default=0.0,
                             help='Test split ratio for single input file (default: 0.0)')

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('-m', '--model-path', type=str,
                             help='Local model path (takes precedence over --hf-model-name)')
    model_group.add_argument('--hf-model-name', default='bert-base-uncased',
                             help='HuggingFace model name (default: bert-base-uncased)')
    model_group.add_argument('--max-length', type=int, default=512,
                             help='Maximum sequence length (default: 512)')
    model_group.add_argument('--dropout', type=float, default=0.1,
                             help='Dropout rate (default: 0.1)')
    model_group.add_argument('--trust-remote-code', action='store_true',
                             help='Execute model code downloaded from the Hub '
                                  '(needed by Alibaba-NLP/gte-large-en-v1.5)')

    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('-e', '--epochs', '--num-epochs', type=int, default=3,
                                help='Number of training epochs (default: 3)')
    training_group.add_argument('-lr', '--learning-rate', type=float, default=2e-5,
                                help='Learning rate (default: 2e-5)')
    training_group.add_argument('--scheduler', choices=['linear', 'cosine', 'exponential', 'step', 'plateau', 'none'],
                                default='linear', help='Learning rate scheduler (default: linear)')
    training_group.add_argument('--warmup-steps', type=int, default=0,
                                help='Number of warmup steps for linear scheduler (default: 0)')
    training_group.add_argument('--weight-decay', type=float, default=0.01,
                                help='Weight decay (default: 0.01)')
    training_group.add_argument('--patience', '--early-stopping-patience', type=int, default=3,
                                help='Early stopping patience (default: 3)')
    training_group.add_argument('--evaluation-metric', choices=['f1', 'f1_macro', 'f1_weighted', 'accuracy', 'kappa'],
                                default='f1_macro', help='Evaluation metric for early stopping (default: f1_macro)')

    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--batch-size', type=int, default=16,
                            help='Batch size (default: 16)')
    data_group.add_argument('--resample-strategy', choices=['none', 'undersample', 'oversample'],
                            default='none', help='Class balancing strategy (default: none)')
    data_group.add_argument('--use-class-weights', action='store_true',
                            help='Use class weights in loss function')
    data_group.add_argument('--focal-loss', action='store_true',
                            help='Use focal loss for handling class imbalance')
    data_group.add_argument('--focal-alpha', type=float, default=1.0,
                            help='Focal loss alpha parameter (default: 1.0)')
    data_group.add_argument('--focal-gamma', type=float, default=2.0,
                            help='Focal loss gamma parameter (default: 2.0)')
    data_group.add_argument('--label-smoothing', type=float, default=0.0,
                            help='Label smoothing factor (0.0 = no smoothing, default: 0.0)')
    data_group.add_argument('--threshold-moving', action='store_true',
                            help='Use threshold moving for imbalanced classification')
    data_group.add_argument('--weighted-sampling', action='store_true',
                            help='Use weighted random sampling in DataLoader')

    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument('--fp16', action='store_true',
                            help='Enable mixed precision training')
    perf_group.add_argument('--num-workers', type=int, default=4,
                            help='Number of data loader workers (default: 4)')

    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--seed', type=int, default=42,
                             help='Random seed (default: 42)')
    other_group.add_argument('-v', '--verbose', action='store_true',
                             help='Enable verbose logging')

    args = parser.parse_args()

    if args.train_file and not args.val_file:
        parser.error("--val-file is required when using --train-file")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        if args.input:
            logger.info(f"Loading data from {args.input}")
            df = load_dataframe(args.input)

            if args.text_col not in df.columns:
                logger.error(
                    f"Text column '{args.text_col}' not found. Available: {list(df.columns)}")
                sys.exit(1)

            if args.label_col not in df.columns:
                logger.error(
                    f"Label column '{args.label_col}' not found. Available: {list(df.columns)}")
                sys.exit(1)

            texts = df[args.text_col].fillna("").astype(str).tolist()
            raw_labels = df[args.label_col].tolist()

            # Create label mapping - preserve original label names
            unique_labels = sorted(set(raw_labels))
            if all(isinstance(label, (int, np.integer)) for label in unique_labels):
                # If labels are already integers, use them directly
                label_mapping = {str(label): label for label in unique_labels}
            else:
                # If labels are strings, create mapping to integers
                label_mapping = {str(label): i for i,
                                 label in enumerate(unique_labels)}

            labels = [label_mapping[str(label)] for label in raw_labels]
            num_labels = len(unique_labels)

            logger.info(f"Dataset size: {len(texts)}")
            logger.info(f"Number of classes: {num_labels}")
            logger.info(f"Label mapping: {label_mapping}")
            logger.info(
                f"Label distribution: {dict(pd.Series(labels).value_counts().sort_index())}")

            if args.resample_strategy != 'none':
                texts, labels = resample_data(
                    texts, labels, args.resample_strategy, args.seed)

            if args.test_split > 0:
                train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                    texts, labels, test_size=args.test_split, random_state=args.seed, stratify=labels
                )
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    train_val_texts, train_val_labels, test_size=args.val_split,
                    random_state=args.seed, stratify=train_val_labels
                )
            else:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels, test_size=args.val_split, random_state=args.seed, stratify=labels
                )
                test_texts, test_labels = None, None

        else:
            logger.info(f"Loading training data from {args.train_file}")
            train_df = load_dataframe(args.train_file)
            train_texts = train_df[args.text_col].fillna(
                "").astype(str).tolist()
            train_raw_labels = train_df[args.label_col].tolist()

            logger.info(f"Loading validation data from {args.val_file}")
            val_df = load_dataframe(args.val_file)
            val_texts = val_df[args.text_col].fillna("").astype(str).tolist()
            val_raw_labels = val_df[args.label_col].tolist()

            # Combine all labels to create consistent mapping
            all_raw_labels = train_raw_labels + val_raw_labels

            if args.test_file:
                logger.info(f"Loading test data from {args.test_file}")
                test_df = load_dataframe(args.test_file)
                test_texts = test_df[args.text_col].fillna(
                    "").astype(str).tolist()
                test_raw_labels = test_df[args.label_col].tolist()
                all_raw_labels.extend(test_raw_labels)
            else:
                test_texts, test_raw_labels = None, None

            unique_labels = sorted(set(all_raw_labels))
            if all(isinstance(label, (int, np.integer)) for label in unique_labels):
                label_mapping = {str(label): label for label in unique_labels}
            else:
                label_mapping = {str(label): i for i,
                                 label in enumerate(unique_labels)}

            train_labels = [label_mapping[str(label)]
                            for label in train_raw_labels]
            val_labels = [label_mapping[str(label)]
                          for label in val_raw_labels]
            if test_texts:
                test_labels = [label_mapping[str(label)]
                               for label in test_raw_labels]
            else:
                test_labels = None

            num_labels = len(unique_labels)

            logger.info(
                f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
            if test_texts:
                logger.info(f"Test size: {len(test_texts)}")
            logger.info(f"Number of classes: {num_labels}")
            logger.info(f"Label mapping: {label_mapping}")

            # Apply resampling if requested (only on training data)
            if args.resample_strategy != 'none':
                train_texts, train_labels = resample_data(train_texts, train_labels,
                                                          args.resample_strategy, args.seed)

        logger.info(
            f"Final train size: {len(train_texts)}, val size: {len(val_texts)}")

        model_name = args.model_path if args.model_path else args.hf_model_name
        logger.info(f"Loading model: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=args.trust_remote_code)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_name}: {e}")
            logger.info("Trying to load from cache or fallback...")
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            if cache_dir.exists():
                cached_models = [d for d in cache_dir.iterdir(
                ) if 'biomedbert' in d.name.lower()]
                if cached_models:
                    snapshot_dir = cached_models[0] / 'snapshots'
                    if snapshot_dir.exists():
                        snapshots = list(snapshot_dir.iterdir())
                        if snapshots:
                            model_name = str(snapshots[0])
                            logger.info(f"Using cached model: {model_name}")
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                trust_remote_code=args.trust_remote_code)
                        else:
                            raise e
                    else:
                        raise e
                else:
                    raise e
            else:
                raise e

        model = SequenceClassifier(model_name, num_labels, args.dropout,
                                   trust_remote_code=args.trust_remote_code)
        model.to(device)

        class_weights = None
        if args.use_class_weights:
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(train_labels), y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(device)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")

        loss_function = get_loss_function(args, class_weights, device)
        if loss_function is not None:
            loss_function = loss_function.to(device)
            logger.info("Using custom loss function")
        else:
            logger.info(
                "Using model's built-in loss function (original behavior)")

        threshold_mover = None
        if args.threshold_moving:
            logger.info("Threshold moving will be applied during evaluation")
            threshold_mover = ThresholdMoving(train_labels)
        else:
            # Ensure threshold_mover is None for default behavior
            threshold_mover = None

        train_dataset = TextDataset(
            train_texts, train_labels, tokenizer, args.max_length)
        val_dataset = TextDataset(
            val_texts, val_labels, tokenizer, args.max_length)

        train_sampler = None
        shuffle = True
        if args.weighted_sampling:
            logger.info("Using weighted random sampling for training")
            train_sampler = create_weighted_sampler(train_labels)
            shuffle = False  # Don't shuffle when using sampler

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=shuffle, sampler=train_sampler,
                                      num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)

        if test_texts:
            test_dataset = TextDataset(
                test_texts, test_labels, tokenizer, args.max_length)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)

        num_training_steps = len(train_dataloader) * args.epochs
        scheduler = get_scheduler_fn(args.scheduler, optimizer, num_training_steps,
                                     warmup_steps=args.warmup_steps)

        best_metric = 0
        patience_counter = 0

        args.output.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")
        logger.info(
            f"Using '{args.evaluation_metric}' metric for early stopping")

        for epoch in range(args.epochs):
            if killer.kill_now:
                logger.info("Training interrupted by user")
                break

            logger.info(f"Epoch {epoch + 1}/{args.epochs}")

            train_loss = train_epoch(model, train_dataloader, optimizer, scheduler,
                                     device, args.fp16, loss_function)

            val_predictions, val_labels_true, val_loss = evaluate(model, val_dataloader,
                                                                  device, args.fp16, threshold_mover)

            metrics = compute_metrics(val_predictions, val_labels_true)

            # Get target metric for early stopping - map argument names to metric keys
            metric_mapping = {
                'f1': 'f1_macro',  # Default f1 to f1_macro for backward compatibility
                'f1_macro': 'f1_macro',
                'f1_weighted': 'f1_weighted',
                'f1_micro': 'f1_micro',
                'accuracy': 'accuracy',
                'kappa': 'cohen_kappa'
            }

            metric_key = metric_mapping.get(args.evaluation_metric, 'f1_macro')
            current_metric = metrics.get(metric_key, metrics['f1_macro'])

            logger.info(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {metrics['accuracy']:.4f}, "
                        f"Val F1 Macro: {metrics['f1_macro']:.4f}, "
                        f"Val F1 Weighted: {metrics['f1_weighted']:.4f}")
            if args.evaluation_metric == 'kappa':
                logger.info(f"Val Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
            logger.info(
                f"Current evaluation metric ({args.evaluation_metric}): {current_metric:.4f}")

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_metric)

            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0

                test_metrics = None
                if test_texts:
                    test_predictions, test_labels_true, test_loss = evaluate(
                        model, test_dataloader, device, args.fp16, threshold_mover
                    )
                    test_metrics = compute_metrics(
                        test_predictions, test_labels_true)
                    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
                                f"Test F1 Macro: {test_metrics['f1_macro']:.4f}")

                results = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metrics': metrics,
                    'test_metrics': test_metrics,
                    'best_metric': best_metric,
                    'evaluation_metric': args.evaluation_metric
                }

                save_complete_model(model, tokenizer, args,
                                    results, label_mapping, args.output)

                # Save basic results.json for backward compatibility
                basic_results = {
                    'epoch': epoch + 1,
                    'metrics': test_metrics if test_metrics else metrics,
                    'label_map': {v: k for k, v in label_mapping.items()},
                    'args': vars(args)
                }

                with open(args.output / 'results.json', 'w') as f:
                    json.dump(basic_results, f, indent=2, default=str)

                logger.info(
                    f"New best {args.evaluation_metric}: {current_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{args.patience}")

                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info(
            f"Training completed. Best {args.evaluation_metric}: {best_metric:.4f}")
        logger.info(f"Model and results saved to {args.output}")

        logger.info("Loading best model for final evaluation...")

        # Load best model weights into the existing model (don't create new instance)
        best_state_dict = torch.load(
            args.output / 'best_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(best_state_dict)
        model.eval()  # Set to evaluation mode

        final_val_predictions, final_val_labels, _ = evaluate(
            model, val_dataloader, device, args.fp16, threshold_mover)

        final_test_predictions, final_test_labels = None, None
        if test_texts:
            final_test_predictions, final_test_labels, _ = evaluate(
                model, test_dataloader, device, args.fp16, threshold_mover)

        reverse_label_map = {v: k for k, v in label_mapping.items()}
        target_names = [str(reverse_label_map[i]) for i in range(num_labels)]

        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print("\nFinal Validation Classification Report:")
        print(classification_report(final_val_labels, final_val_predictions,
                                    target_names=target_names, zero_division=0))

        if final_test_predictions is not None:
            print("\nFinal Test Classification Report:")
            print(classification_report(final_test_labels, final_test_predictions,
                                        target_names=target_names, zero_division=0))

        print("="*60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
