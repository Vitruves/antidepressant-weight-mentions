#!/usr/bin/env python3
"""
Advanced BERT Training Pipeline - Complete Feature Set with Modern Optimizations
Supports all BERT variants with state-of-the-art features and comprehensive configuration
"""

import os
import sys
import json
import warnings
import logging
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import time
import transformers

# Custom warning filter for specific messages
class DatasetLengthWarningFilter(logging.Filter):
    def filter(self, record):
        # Filter out the specific dataset length warning and other noisy messages
        message = record.getMessage()
        return not any(phrase in message for phrase in [
            "The used dataset had no length",
            "returning gathered tensors",
            "You should drop the remainder yourself"
        ])

# Disable tokenizer warnings early
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Defer heavy imports until needed
_modules = {}

# Global trainer instance for signal handling
_global_trainer = None

def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully by saving best model"""
    global _global_trainer
    if _global_trainer is not None:
        logger = _global_trainer.logger
        logger.warning("Received interrupt signal - saving best model checkpoint")
        try:
            _global_trainer._save_best_checkpoint_on_interrupt()
        except Exception as e:
            logger.error(f"Failed to save best model checkpoint: {str(e)}")
    sys.exit(1)

def get_module(name: str):
    """Lazy import modules to speed up CLI help and avoid import conflicts"""
    if name not in _modules:
        if name == 'torch':
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            _modules[name] = torch
            _modules['nn'] = nn
            _modules['F'] = F
        elif name == 'transformers':
            import transformers
            _modules[name] = transformers
        elif name == 'datasets':
            import datasets
            _modules[name] = datasets
        elif name == 'sklearn':
            import sklearn
            from sklearn import metrics, model_selection
            _modules[name] = sklearn
            _modules['metrics'] = metrics
            _modules['model_selection'] = model_selection
        elif name == 'pandas':
            import pandas as pd
            _modules[name] = pd
        elif name == 'numpy':
            import numpy as np
            _modules[name] = np
        elif name == 'tqdm':
            import tqdm
            _modules[name] = tqdm
        elif name == 'imblearn':
            try:
                import imblearn
                from imblearn.over_sampling import RandomOverSampler
                from imblearn.under_sampling import RandomUnderSampler
                _modules[name] = imblearn
                _modules['RandomOverSampler'] = RandomOverSampler
                _modules['RandomUnderSampler'] = RandomUnderSampler
            except ImportError:
                _modules[name] = None
    return _modules[name]

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored log levels"""
    COLORS = {
        'DEBUG': '\033[93m',    # Yellow
        'INFO': '\033[94m',     # Blue  
        'WARNING': '\033[91m',  # Red
        'ERROR': '\033[91m',    # Red
        'SUCCESS': '\033[92m',  # Green
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        timestamp = datetime.now().strftime("%H:%M")
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        return f"{timestamp} - {color}{record.levelname}{reset}: {record.getMessage()}"

class TqdmLoggingHandler(logging.Handler):
    """Handler that routes logs through tqdm to maintain bottom-sticky progress bar"""
    
    def __init__(self, progress_bar=None):
        super().__init__()
        self.progress_bar = progress_bar
        
    def emit(self, record):
        try:
            msg = self.format(record)
            # Always use tqdm.write to preserve progress bar position
            tqdm = get_module('tqdm')
            if tqdm:
                # Use tqdm.write() to output above progress bars
                tqdm.tqdm.write(msg, file=sys.stdout)
                # Force refresh of any active progress bars
                if self.progress_bar and hasattr(self.progress_bar, 'refresh'):
                    self.progress_bar.refresh()
            else:
                print(msg, file=sys.stdout, flush=True)
        except Exception:
            self.handleError(record)

def setup_logging(verbose: bool = False, debug: bool = False, progress_bar=None):
    """Setup colored logging with SUCCESS level and tqdm integration"""
    
    # Add SUCCESS log level
    if not hasattr(logging, 'SUCCESS'):
        logging.SUCCESS = 25
        logging.addLevelName(logging.SUCCESS, 'SUCCESS')
        
        def success(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.SUCCESS):
                self._log(logging.SUCCESS, message, args, **kwargs)
        logging.Logger.success = success
    
    # Set log level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    # Setup handler
    handler = TqdmLoggingHandler(progress_bar)
    handler.setFormatter(ColoredFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Add custom filter to root logger
    root_logger.addFilter(DatasetLengthWarningFilter())
    
    # Configure transformers logging to go through our handler
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.handlers.clear()
    transformers_logger.addHandler(handler)
    transformers_logger.setLevel(logging.WARNING)  # Allow warnings but suppress info
    
    # Silence noisy libraries
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    
    # Specifically suppress the dataset length warning
    logging.getLogger('transformers.data.data_collator').setLevel(logging.ERROR)
    
    # Apply filter to specific loggers that might emit the warning
    for logger_name in ['accelerate', 'accelerate.accelerator', 'transformers.trainer']:
        logger = logging.getLogger(logger_name)
        logger.addFilter(DatasetLengthWarningFilter())
    
    return logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Comprehensive training configuration with all original features"""
    
    # Model architecture
    model_name: str = "answerdotai/ModernBERT-base"
    task_type: str = "classification"  # "classification" or "regression"
    num_labels: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    
    # Custom head configuration
    custom_head_layers: List[int] = field(default_factory=lambda: [])
    custom_head_dropout: float = 0.1
    custom_head_activation: str = "gelu"
    pooling_strategy: str = "cls"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    max_epochs: int = 3
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Batch sizes and data loading
    train_batch_size: int = 16
    eval_batch_size: int = 32
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Precision and performance
    fp16: bool = False
    bf16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = False
    dataloader_prefetch_factor: int = 2
    
    # Transfer learning
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = 0
    unfreeze_schedule: Optional[List[int]] = None
    progressive_unfreezing: bool = False
    layer_wise_lr_decay: float = 1.0
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Loss function configuration
    loss_type: str = "cross_entropy"
    class_weights: Optional[List[float]] = None
    auto_class_weights: bool = False
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Data resampling
    resampling_strategy: str = "none"
    oversample_ratio: float = 1.0
    undersample_ratio: float = 1.0
    
    # Data augmentation
    augmentation_strategy: str = "none"
    augmentation_prob: float = 0.5
    synonym_replacement_prob: float = 0.1
    random_insertion_prob: float = 0.1
    random_swap_prob: float = 0.1
    random_deletion_prob: float = 0.1
    
    # Evaluation and checkpointing
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False  # Note: Set to True when using metrics like accuracy, f1, or kappa
    
    # Overfitting detection
    enable_overfitting_detection: bool = True
    overfitting_threshold: float = 0.05  # Gap between train and val metrics
    overfitting_patience: int = 3  # Number of evaluations before declaring overfit
    overfitting_metric: str = "loss"  # Metric to monitor (loss, accuracy, f1)
    overfitting_relative_threshold: float = 0.1  # Relative difference threshold (10%)
    overfitting_action: str = "warn"  # "warn", "stop", or "reduce_lr"
    
    # Logging and monitoring
    logging_steps: int = 50
    log_level: str = "error"
    report_to: List[str] = field(default_factory=lambda: [])
    run_name: Optional[str] = None
    
    # Data configuration
    max_seq_length: int = 512
    text_column: str = "text"
    label_column: str = "label"
    validation_split_size: float = 0.2
    test_split_size: float = 0.1
    stratify_split: bool = True
    
    # Distributed training
    local_rank: int = -1
    distributed_backend: str = "nccl"
    world_size: int = 1
    
    # Output configuration
    output_dir: str = "./results"
    overwrite_output_dir: bool = False
    resume_from_checkpoint: Optional[str] = None
    save_safetensors: bool = True
    
    # Hub integration
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
    
    # Hardware configuration
    device: str = "auto"
    gpu_id: Optional[int] = None
    no_dataparallel: bool = False
    flash_attention: bool = True
    no_flash_attention: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and auto-configuration"""
        # Validation split constraints
        if self.validation_split_size < 0.05 or self.validation_split_size > 0.4:
            raise ValueError("validation_split_size must be between 0.05 and 0.4")
        if self.test_split_size < 0 or self.test_split_size > 0.3:
            raise ValueError("test_split_size must be between 0 and 0.3")
        
        # Auto-adjust greater_is_better based on metric
        metrics_where_higher_is_better = ["accuracy", "f1", "precision", "recall", "kappa", "auc", "roc_auc"]
        if any(metric in self.metric_for_best_model for metric in metrics_where_higher_is_better):
            self.greater_is_better = True
            # Use print instead of logging since this happens during config creation
            print(f"Auto-setting greater_is_better=True for metric: {self.metric_for_best_model}")
        
        # Ensure save_strategy matches eval_strategy for proper best model tracking
        if self.eval_strategy == "epoch" and self.save_strategy == "steps":
            self.save_strategy = "epoch"
            print(f"Auto-setting save_strategy=epoch to match eval_strategy for proper best model tracking")
        elif self.eval_strategy == "steps" and self.save_strategy == "epoch":
            self.save_strategy = "steps"
            print(f"Auto-setting save_strategy=steps to match eval_strategy for proper best model tracking")


class AdvancedBERTTrainer:
    """Complete BERT training pipeline with all original features"""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.datasets = {}
        self.progress_bar = None
        self._original_train_dataset = None
        
        # Overfitting detection state
        self.overfitting_counter = 0
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.is_overfitting = False
        
        # Initialize components
        self._setup_device()
        self._setup_output_directory()
    
    def _setup_device(self):
        """Comprehensive device configuration with multi-GPU support"""
        torch = get_module('torch')
        
        self.logger.info("Initializing compute device configuration")
        
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                self.device = torch.device("cuda")
                self.logger.info(f"Auto-detected CUDA with {device_count} GPU(s)")
                
                # Display GPU information
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.logger.info("Auto-detected Apple Silicon MPS device")
            else:
                self.device = torch.device("cpu")
                self.logger.warning("No GPU available, using CPU (will be slow)")
                
        elif self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.device = torch.device("cuda")
            
        elif self.config.device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested but not available")
            self.device = torch.device("mps")
            
        elif self.config.device == "cpu":
            self.device = torch.device("cpu")
            self.logger.warning("Using CPU device (training will be slow)")
        else:
            self.device = torch.device(self.config.device)
        
        # Handle specific GPU selection
        if self.device.type == "cuda" and self.config.gpu_id is not None:
            gpu_count = torch.cuda.device_count()
            if self.config.gpu_id >= gpu_count:
                raise ValueError(f"GPU {self.config.gpu_id} not available (only {gpu_count} GPUs)")
            self.device = torch.device(f"cuda:{self.config.gpu_id}")
            torch.cuda.set_device(self.device)
            self.logger.info(f"Using specific GPU {self.config.gpu_id}")
        
        # Handle distributed training
        elif self.device.type == "cuda" and self.config.local_rank != -1:
            self.logger.info("Configuring distributed training")
            torch.distributed.init_process_group(backend=self.config.distributed_backend)
            self.device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(self.device)
        elif self.device.type == "cuda" and torch.cuda.device_count() > 1 and not self.config.no_dataparallel:
            self.logger.info(f"Multi-GPU configuration detected - {torch.cuda.device_count()} GPUs available")
        elif self.device.type == "cuda" and self.config.no_dataparallel:
            self.logger.info("DataParallel disabled - using single GPU mode")
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        
        self.logger.success(f"Device configuration completed - using {self.device}")
    
    def _setup_output_directory(self):
        """Create comprehensive output directory structure"""
        self.logger.info("Setting up output directory structure")
        
        output_path = Path(self.config.output_dir)
        
        if output_path.exists() and not self.config.overwrite_output_dir:
            if any(output_path.iterdir()):
                raise ValueError(f"Output directory {output_path} exists and is not empty - use --overwrite-output to overwrite")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["checkpoints", "logs", "predictions", "metrics", "plots"]
        for subdir in subdirs:
            (output_path / subdir).mkdir(exist_ok=True)
        
        self.logger.success(f"Output directory structure created at {output_path}")
    
    def _setup_progress_bar(self, total_steps: int, desc: str = "Training"):
        """Setup bottom-sticky progress bar with proper tqdm integration"""
        tqdm = get_module('tqdm')
        
        if self.progress_bar:
            self.progress_bar.close()
        
        # Ensure we're starting fresh
        if hasattr(tqdm.tqdm, '_instances'):
            tqdm.tqdm._instances.clear()
        
        self.progress_bar = tqdm.tqdm(
            total=total_steps,
            desc=f"\033[1;36m{desc}\033[0m",
            unit="step",
            leave=True,   # Keep the bar visible for better control
            position=0,   # Bottom position
            dynamic_ncols=True,
            colour='green',
            bar_format='{l_bar}\033[32m{bar}\033[0m| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            file=sys.stdout,
            mininterval=0.1,
            smoothing=0.3,
            ascii=False,
            ncols=None,  # Auto-detect terminal width
        )
        
        # Store original stdout to restore later if needed
        if not hasattr(self, '_original_stdout'):
            self._original_stdout = sys.stdout
        
        # Initialize epoch progress bar (will be created per epoch)
        self.epoch_progress_bar = None
        
        # Update logging handler to use this progress bar
        for handler in logging.getLogger().handlers:
            if isinstance(handler, TqdmLoggingHandler):
                handler.progress_bar = self.progress_bar
    
    def load_data(self, train_file: Optional[str] = None, val_file: Optional[str] = None,
                  test_file: Optional[str] = None, input_file: Optional[str] = None):
        """Comprehensive data loading with all preprocessing features"""
        
        self.logger.info("Starting comprehensive data loading pipeline")
        
        try:
            if input_file:
                self.logger.info("Processing single file with automatic splitting")
                dataset = self._load_single_file(input_file)
                datasets_dict = self._split_dataset(dataset)
            else:
                self.logger.info("Processing pre-split dataset files")
                datasets_dict = self._load_separate_files(train_file, val_file, test_file)
            
            # Store original training dataset for class weight computation
            if 'train' in datasets_dict and datasets_dict['train'] is not None:
                self._original_train_dataset = datasets_dict['train']
                self._detect_and_update_labels()
            
            # Apply preprocessing pipeline
            self.logger.info("Applying preprocessing pipeline to all datasets")
            processed_datasets = {}
            
            for split, dataset in datasets_dict.items():
                if dataset is not None:
                    self.logger.info(f"Processing {split} dataset - {len(dataset)} samples")
                    processed_datasets[split] = self._preprocess_dataset(dataset, split)
                    self.logger.info(f"{split} dataset processing completed")
            
            self.datasets = processed_datasets
            self.logger.success("Data loading pipeline completed successfully")
            return processed_datasets
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _load_single_file(self, filepath: str):
        """Load single file with comprehensive format support"""
        datasets = get_module('datasets')
        pd = get_module('pandas')
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        self.logger.info(f"Loading dataset from {filepath}")
        
        try:
            if filepath.suffix.lower() == '.xlsx':
                df = pd.read_excel(filepath)
            elif filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() == '.parquet':
                df = pd.read_parquet(filepath)
            elif filepath.suffix.lower() == '.json':
                df = pd.read_json(filepath, lines=True)
            elif filepath.suffix.lower() == '.tsv':
                df = pd.read_csv(filepath, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            # Validate required columns
            if self.config.text_column not in df.columns:
                raise ValueError(f"Text column '{self.config.text_column}' not found in dataset columns: {list(df.columns)}")
            if self.config.label_column not in df.columns:
                raise ValueError(f"Label column '{self.config.label_column}' not found in dataset columns: {list(df.columns)}")
            
            dataset = datasets.Dataset.from_pandas(df)
            self.logger.success(f"Successfully loaded {len(dataset)} samples from {filepath}")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {str(e)}")
            raise
    
    def _split_dataset(self, dataset):
        """Split dataset with stratification support"""
        model_selection = get_module('model_selection')
        
        self.logger.info("Executing dataset splitting strategy")
        
        total_size = len(dataset)
        test_size = int(total_size * self.config.test_split_size) if self.config.test_split_size > 0 else 0
        val_size = int((total_size - test_size) * self.config.validation_split_size)
        train_size = total_size - val_size - test_size
        
        self.logger.info(f"Split configuration: Train={train_size}, Validation={val_size}, Test={test_size}")
        
        if self.config.stratify_split and self.config.label_column in dataset.column_names:
            self.logger.info("Applying stratified splitting to preserve label distribution")
            labels = dataset[self.config.label_column]
            
            if test_size > 0:
                train_val_indices, test_indices = model_selection.train_test_split(
                    list(range(len(dataset))),
                    test_size=test_size,
                    stratify=labels,
                    random_state=42
                )
                test_dataset = dataset.select(test_indices)
                train_val_dataset = dataset.select(train_val_indices)
                train_val_labels = [labels[i] for i in train_val_indices]
            else:
                test_dataset = None
                train_val_dataset = dataset
                train_val_labels = labels
            
            if val_size > 0:
                train_indices, val_indices = model_selection.train_test_split(
                    list(range(len(train_val_dataset))),
                    test_size=val_size,
                    stratify=train_val_labels,
                    random_state=42
                )
                train_dataset = train_val_dataset.select(train_indices)
                val_dataset = train_val_dataset.select(val_indices)
            else:
                train_dataset = train_val_dataset
                val_dataset = None
        else:
            self.logger.info("Applying random splitting strategy")
            if test_size > 0:
                dataset_split = dataset.train_test_split(test_size=test_size, seed=42)
                test_dataset = dataset_split['test']
                train_val_dataset = dataset_split['train']
            else:
                test_dataset = None
                train_val_dataset = dataset
            
            if val_size > 0:
                train_val_split = train_val_dataset.train_test_split(
                    test_size=val_size / len(train_val_dataset), seed=42
                )
                train_dataset = train_val_split['train']
                val_dataset = train_val_split['test']
            else:
                train_dataset = train_val_dataset
                val_dataset = None
        
        self.logger.success("Dataset splitting operation completed")
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }
    
    def _load_separate_files(self, train_file: str, val_file: Optional[str] = None,
                           test_file: Optional[str] = None):
        """Load separate dataset files"""
        datasets_dict = {}
        
        if train_file:
            self.logger.info("Loading training dataset file")
            datasets_dict['train'] = self._load_single_file(train_file)
        
        if val_file:
            self.logger.info("Loading validation dataset file")
            datasets_dict['validation'] = self._load_single_file(val_file)
        
        if test_file:
            self.logger.info("Loading test dataset file")
            datasets_dict['test'] = self._load_single_file(test_file)
            
        return datasets_dict
    
    def _detect_and_update_labels(self):
        """Auto-detect number of labels and validate encoding"""
        if not self._original_train_dataset:
            return
        
        # Skip label detection for regression tasks
        if self.config.task_type == "regression":
            self.logger.info("Regression task detected - skipping label detection")
            return
        
        labels = self._original_train_dataset[self.config.label_column]
        unique_labels = sorted(set(labels))
        detected_num_labels = len(unique_labels)
        
        # Enhanced logging for label distribution
        label_counts = {label: labels.count(label) for label in unique_labels}
        self.logger.info(f"Label distribution: {label_counts}")
        
        # Check if labels need mapping to 0-indexed consecutive integers
        expected_labels = list(range(detected_num_labels))
        if unique_labels != expected_labels:
            self.logger.warning(f"⚠️  Labels need mapping: {unique_labels} -> {expected_labels}")
            self.logger.warning("Original labels are not 0-indexed consecutive integers. This may indicate:")
            self.logger.warning("  - Labels starting from 1 instead of 0")
            self.logger.warning("  - Non-consecutive label values")
            self.logger.warning("  - String labels that need encoding")
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
            self.logger.info(f"Created label mapping: {self.label_map}")
        else:
            self.label_map = None
            self.reverse_label_map = None
            self.logger.info("✓ Labels already properly encoded (0-indexed consecutive)")
        
        if detected_num_labels != self.config.num_labels:
            self.logger.warning(f"⚠️  Auto-detected {detected_num_labels} classes in dataset: {unique_labels}")
            self.logger.warning(f"Updating num_labels from {self.config.num_labels} to {detected_num_labels}")
            self.config.num_labels = detected_num_labels
    
    def _compute_class_weights(self, dataset):
        """Compute class weights for imbalanced datasets"""
        torch = get_module('torch')
        np = get_module('numpy')
        
        self.logger.info("Computing automatic class weights for imbalanced data handling")
        
        labels = dataset[self.config.label_column]
        num_classes = self.config.num_labels
        
        # Apply label mapping if it exists
        if hasattr(self, 'label_map') and self.label_map is not None:
            self.logger.info("Applying label mapping for class weight computation")
            mapped_labels = [self.label_map[label] for label in labels]
            labels = mapped_labels
        
        class_counts = np.bincount(labels, minlength=num_classes)
        total_samples = len(labels)
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        weights = total_samples / (num_classes * class_counts)
        weights = weights / weights.sum() * num_classes
        
        self.logger.info(f"Class weights computed for {num_classes} classes: {weights}")
        return torch.tensor(weights, dtype=torch.float32)
    
    def _apply_resampling(self, dataset, split: str):
        """Apply data resampling strategies"""
        if self.config.resampling_strategy == "none" or split != "train":
            return dataset
        
        np = get_module('numpy')
        
        self.logger.info(f"Applying {self.config.resampling_strategy} resampling strategy to {split} dataset")
        
        labels = np.array(dataset[self.config.label_column])
        indices = np.arange(len(dataset))
        
        if self.config.resampling_strategy == "oversample":
            RandomOverSampler = get_module('RandomOverSampler')
            if RandomOverSampler:
                sampler = RandomOverSampler(sampling_strategy=self.config.oversample_ratio, random_state=42)
                indices_resampled, _ = sampler.fit_resample(indices.reshape(-1, 1), labels)
                indices_resampled = indices_resampled.flatten()
            else:
                self.logger.warning("imbalanced-learn not available, skipping oversampling")
                return dataset
                
        elif self.config.resampling_strategy == "undersample":
            RandomUnderSampler = get_module('RandomUnderSampler')
            if RandomUnderSampler:
                sampler = RandomUnderSampler(sampling_strategy=self.config.undersample_ratio, random_state=42)
                indices_resampled, _ = sampler.fit_resample(indices.reshape(-1, 1), labels)
                indices_resampled = indices_resampled.flatten()
            else:
                self.logger.warning("imbalanced-learn not available, skipping undersampling")
                return dataset
                
        elif self.config.resampling_strategy == "smote":
            self.logger.warning("SMOTE resampling will be applied during training phase")
            return dataset
        
        else:
            return dataset
        
        resampled_dataset = dataset.select(indices_resampled)
        self.logger.success(f"Resampling completed - dataset size changed from {len(dataset)} to {len(resampled_dataset)}")
        
        return resampled_dataset
    
    def _apply_augmentation(self, dataset):
        """Apply text augmentation techniques"""
        if self.config.augmentation_strategy == "none":
            return dataset
            
        np = get_module('numpy')
        
        self.logger.info(f"Applying {self.config.augmentation_strategy} text augmentation techniques")
        
        def augment_text(example):
            text = example[self.config.text_column]
            
            if np.random.random() < self.config.augmentation_prob:
                if self.config.synonym_replacement_prob > 0 and np.random.random() < self.config.synonym_replacement_prob:
                    text = self._synonym_replacement(text)
                
                if self.config.random_insertion_prob > 0 and np.random.random() < self.config.random_insertion_prob:
                    text = self._random_insertion(text)
                
                if self.config.random_swap_prob > 0 and np.random.random() < self.config.random_swap_prob:
                    text = self._random_swap(text)
                
                if self.config.random_deletion_prob > 0 and np.random.random() < self.config.random_deletion_prob:
                    text = self._random_deletion(text)
            
            example[self.config.text_column] = text
            return example
        
        augmented_dataset = dataset.map(augment_text, desc="Augmenting text")
        self.logger.success("Text augmentation pipeline completed")
        return augmented_dataset
    
    def _synonym_replacement(self, text: str, n: int = 1) -> str:
        """Simple synonym replacement (placeholder implementation)"""
        words = text.split()
        if len(words) < 2:
            return text
        
        np = get_module('numpy')
        for _ in range(n):
            if len(words) > 0:
                idx = np.random.randint(0, len(words))
                # Simple replacement strategy
                words[idx] = words[idx] + "_syn"
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str, n: int = 1) -> str:
        """Random word insertion"""
        np = get_module('numpy')
        words = text.split()
        for _ in range(n):
            if len(words) > 0:
                idx = np.random.randint(0, len(words) + 1)
                words.insert(idx, 'the')
        return ' '.join(words)
    
    def _random_swap(self, text: str, n: int = 1) -> str:
        """Random word swap"""
        np = get_module('numpy')
        words = text.split()
        for _ in range(n):
            if len(words) > 1:
                idx1 = np.random.randint(0, len(words))
                idx2 = np.random.randint(0, len(words))
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Random word deletion"""
        np = get_module('numpy')
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if np.random.random() > p]
        if len(new_words) == 0:
            return words[np.random.randint(0, len(words))]
        return ' '.join(new_words)
    
    def _preprocess_dataset(self, dataset, split: str):
        """Comprehensive dataset preprocessing"""
        datasets_module = get_module('datasets')
        
        self.logger.info(f"Starting preprocessing pipeline for {split} dataset")
        
        # Apply resampling
        dataset = self._apply_resampling(dataset, split)
        
        # Apply augmentation for training data
        if self.config.augmentation_strategy != "none" and split == "train":
            dataset = self._apply_augmentation(dataset)
        
        # Tokenization
        self.logger.info(f"Tokenizing {split} dataset with max sequence length {self.config.max_seq_length}")
        
        def tokenize_function(examples):
            texts = examples[self.config.text_column]
            
            processed_texts = []
            for text in texts:
                if text is None:
                    processed_texts.append("")
                elif isinstance(text, str):
                    processed_texts.append(text)
                else:
                    processed_texts.append(str(text))
            
            tokenized = self.tokenizer(
                processed_texts,
                truncation=True,
                padding=False,  # Let DataCollatorWithPadding handle dynamic padding
                max_length=self.config.max_seq_length,
                return_tensors=None
            )
            
            if self.config.label_column in examples:
                labels = examples[self.config.label_column]
                # Apply label mapping if it exists
                if hasattr(self, 'label_map') and self.label_map is not None:
                    labels = [self.label_map[label] for label in labels]
                tokenized["labels"] = labels
            
            return tokenized
        
        datasets_module.disable_progress_bar()
        
        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
            
            # Force dataset materialization by accessing its length
            # This prevents the "dataset has no length" warning during training
            try:
                dataset_length = len(tokenized_dataset)
                self.logger.info(f"Dataset materialized with {dataset_length} samples")
            except (TypeError, AttributeError):
                # If dataset doesn't have __len__, force it to be concrete
                tokenized_dataset = tokenized_dataset.with_format(None)
                dataset_length = len(tokenized_dataset)
                self.logger.info(f"Dataset forced to concrete format with {dataset_length} samples")
            
        finally:
            datasets_module.enable_progress_bar()
        
        format_columns = ['input_ids', 'attention_mask']
        if 'labels' in tokenized_dataset.column_names:
            format_columns.append('labels')
        tokenized_dataset.set_format(type='torch', columns=format_columns)
        
        self.logger.success(f"Preprocessing pipeline completed for {split} dataset")
        return tokenized_dataset
    
    def setup_tokenizer(self):
        """Initialize tokenizer with comprehensive configuration"""
        transformers = get_module('transformers')
        
        self.logger.info(f"Initializing tokenizer from {self.config.model_name}")
        
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("Added padding token to tokenizer configuration")
                
            self.logger.success("Tokenizer initialization completed")
            
        except Exception as e:
            self.logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise
    
    def setup_model(self):
        """Comprehensive model setup with all features"""
        transformers = get_module('transformers')
        torch = get_module('torch')
        
        if self.config.task_type == "regression":
            self.logger.info(f"Initializing model {self.config.model_name} for regression task")
        else:
            self.logger.info(f"Initializing model {self.config.model_name} for {self.config.num_labels} classes")
        
        try:
            # For regression, num_labels should be 1
            num_labels = 1 if self.config.task_type == "regression" else self.config.num_labels
            
            model_config = transformers.AutoConfig.from_pretrained(
                self.config.model_name,
                num_labels=num_labels,
                hidden_dropout_prob=self.config.hidden_dropout_prob,
                attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            )
            
            # Update max_position_embeddings if using longer sequences
            if self.config.max_seq_length > model_config.max_position_embeddings:
                self.logger.warning(f"Configured max_seq_length ({self.config.max_seq_length}) exceeds model's default max_position_embeddings ({model_config.max_position_embeddings})")
                self.logger.info(f"Updating max_position_embeddings to {self.config.max_seq_length}")
                model_config.max_position_embeddings = self.config.max_seq_length
            
            model_kwargs = {
                "config": model_config,
                # Note: torch_dtype removed to let AMP handle mixed precision properly
            }
            
            # Flash Attention configuration
            if (self.device.type == "cuda" and 
                self.config.flash_attention and 
                not self.config.no_flash_attention):
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    # Note: device_map removed to avoid conflicts with manual .to() call
                    self.logger.info("Enabling Flash Attention 2.0 for optimized performance")
                except ImportError:
                    self.logger.warning("Flash Attention not available, using standard attention")
                except Exception as e:
                    self.logger.warning(f"Flash Attention setup failed: {e}")
            
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            if not (hasattr(self.model, 'device') and str(self.model.device) == str(self.device)):
                self.model.to(self.device)
            
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Resize position embeddings if needed
            if self.config.max_seq_length > 512:
                self._resize_position_embeddings()
            
            self._apply_custom_model_config()
            self._apply_layer_freezing()
            
            # DataParallel configuration
            if (torch.cuda.device_count() > 1 and 
                self.config.local_rank == -1 and 
                not self.config.no_dataparallel and 
                self.config.gpu_id is None):
                self.model = torch.nn.DataParallel(self.model)
                self.logger.info(f"Applied DataParallel across {torch.cuda.device_count()} GPUs")
            else:
                self.logger.info("Using single device training configuration")
            
            self.logger.success("Model initialization and configuration completed")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def _apply_custom_model_config(self):
        """Apply custom head configuration"""
        if self.config.custom_head_layers:
            torch = get_module('torch')
            self.logger.info("Applying custom classification head architecture")
            
            if hasattr(self.model, 'classifier'):
                hidden_size = self.model.classifier.in_features
                layers = []
                prev_size = hidden_size
                
                for layer_size in self.config.custom_head_layers:
                    layers.extend([
                        torch.nn.Linear(prev_size, layer_size),
                        torch.nn.Dropout(self.config.custom_head_dropout),
                        self._get_activation_function(self.config.custom_head_activation)
                    ])
                    prev_size = layer_size
                
                # For regression, output dimension is 1
                output_dim = 1 if self.config.task_type == "regression" else self.config.num_labels
                layers.append(torch.nn.Linear(prev_size, output_dim))
                
                self.model.classifier = torch.nn.Sequential(*layers)
                self.logger.success("Custom classification head architecture applied")
    
    def _resize_position_embeddings(self):
        """Resize position embeddings to handle longer sequences"""
        torch = get_module('torch')
        nn = get_module('nn')
        
        self.logger.info(f"Resizing position embeddings to support max_seq_length={self.config.max_seq_length}")
        
        # Get the base model (handles different architectures)
        if hasattr(self.model, 'bert'):
            base_model = self.model.bert
        elif hasattr(self.model, 'roberta'):
            base_model = self.model.roberta
        elif hasattr(self.model, 'distilbert'):
            base_model = self.model.distilbert
        else:
            self.logger.warning("Cannot identify base model architecture for position embedding resizing")
            return
        
        # Get current embeddings
        old_embeddings = base_model.embeddings.position_embeddings
        old_num_pos = old_embeddings.weight.size(0)
        new_num_pos = self.config.max_seq_length + 2  # Account for special tokens
        
        if new_num_pos <= old_num_pos:
            self.logger.info(f"No position embedding resize needed: {new_num_pos} <= {old_num_pos}")
            return
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_pos, old_embeddings.weight.size(1))
        
        # Initialize new embeddings
        nn.init.normal_(new_embeddings.weight, std=0.02)
        
        # Copy old embeddings
        with torch.no_grad():
            new_embeddings.weight[:old_num_pos] = old_embeddings.weight
        
        # Replace embeddings
        base_model.embeddings.position_embeddings = new_embeddings
        base_model.embeddings.register_buffer(
            "position_ids", torch.arange(new_num_pos).expand((1, -1)), persistent=False
        )
        
        # Update config
        base_model.config.max_position_embeddings = new_num_pos
        
        self.logger.success(f"Position embeddings resized from {old_num_pos} to {new_num_pos}")
    
    def _get_activation_function(self, activation: str):
        """Get activation function by name"""
        torch = get_module('torch')
        activations = {
            'relu': torch.nn.ReLU(),
            'gelu': torch.nn.GELU(),
            'tanh': torch.nn.Tanh(),
            'silu': torch.nn.SiLU(),
            'leaky_relu': torch.nn.LeakyReLU(),
        }
        return activations.get(activation.lower(), torch.nn.GELU())
    
    def _apply_layer_freezing(self):
        """Apply layer freezing strategies"""
        if self.config.freeze_embeddings:
            self.logger.info("Applying embedding layer freezing")
            if hasattr(self.model, 'bert'):
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(self.model, 'roberta'):
                for param in self.model.roberta.embeddings.parameters():
                    param.requires_grad = False
            self.logger.success("Embedding layers frozen")
        
        if self.config.freeze_encoder_layers > 0:
            self.logger.info(f"Freezing bottom {self.config.freeze_encoder_layers} encoder layers")
            if hasattr(self.model, 'bert'):
                encoder_layers = self.model.bert.encoder.layer
            elif hasattr(self.model, 'roberta'):
                encoder_layers = self.model.roberta.encoder.layer
            else:
                self.logger.warning("Cannot identify encoder layers for freezing")
                return
            
            for i in range(min(self.config.freeze_encoder_layers, len(encoder_layers))):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
            self.logger.success(f"Frozen {min(self.config.freeze_encoder_layers, len(encoder_layers))} encoder layers")
    
    def _setup_optimizer_and_scheduler(self):
        """Comprehensive optimizer and scheduler setup"""
        transformers = get_module('transformers')
        torch = get_module('torch')
        
        self.logger.info("Configuring optimizer and learning rate scheduler")
        
        no_decay = ["bias", "LayerNorm.weight"]
        
        if self.config.layer_wise_lr_decay < 1.0:
            self.logger.info("Applying layer-wise learning rate decay strategy")
            optimizer_grouped_parameters = self._get_layer_wise_parameters()
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                             if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                             if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
        
        # Optimizer selection
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Calculate training steps
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.datasets['train']) // (
                self.config.train_batch_size * self.config.gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * self.config.max_epochs
        
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.logger.info(f"Training schedule configured - total steps: {total_steps}, warmup steps: {warmup_steps}")
        
        # Scheduler selection
        if self.config.scheduler.lower() == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif self.config.scheduler.lower() == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif self.config.scheduler.lower() == "cosine_restarts":
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif self.config.scheduler.lower() == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif self.config.scheduler.lower() == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps
            )
        else:
            scheduler = None
            self.logger.warning(f"Unknown scheduler: {self.config.scheduler}, proceeding without scheduler")
        
        self.logger.success("Optimizer and scheduler configuration completed")
        return optimizer, scheduler
    
    def _get_layer_wise_parameters(self):
        """Get layer-wise parameter groups with learning rate decay"""
        torch = get_module('torch')
        layers = []
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Handle DataParallel wrapper
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        if hasattr(model, 'bert'):
            encoder_layers = model.bert.encoder.layer
            embeddings = model.bert.embeddings
        elif hasattr(model, 'roberta'):
            encoder_layers = model.roberta.encoder.layer
            embeddings = model.roberta.embeddings
        else:
            return [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                             if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                             if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.config.learning_rate,
                },
            ]
        
        # Embeddings with highest decay
        layers.extend([
            {
                "params": [p for n, p in embeddings.named_parameters() 
                         if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate * (self.config.layer_wise_lr_decay ** len(encoder_layers)),
            },
            {
                "params": [p for n, p in embeddings.named_parameters() 
                         if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate * (self.config.layer_wise_lr_decay ** len(encoder_layers)),
            },
        ])
        
        # Encoder layers with decreasing decay
        for i, layer in enumerate(encoder_layers):
            decay_factor = self.config.layer_wise_lr_decay ** (len(encoder_layers) - 1 - i)
            layers.extend([
                {
                    "params": [p for n, p in layer.named_parameters() 
                             if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.learning_rate * decay_factor,
                },
                {
                    "params": [p for n, p in layer.named_parameters() 
                             if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.config.learning_rate * decay_factor,
                },
            ])
        
        # Classifier with full learning rate
        if hasattr(self.model, 'classifier'):
            layers.extend([
                {
                    "params": [p for n, p in self.model.classifier.named_parameters() 
                             if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.classifier.named_parameters() 
                             if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.config.learning_rate,
                },
            ])
        
        return layers
    
    def _compute_metrics(self, eval_pred):
        """Comprehensive metric computation for both classification and regression"""
        np = get_module('numpy')
        get_module('sklearn')  # Ensure sklearn is imported to populate metrics module
        metrics = get_module('metrics')
        
        predictions, labels = eval_pred
        
        if self.config.task_type == "regression":
            # For regression, predictions are already continuous values
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                pred_values = predictions.squeeze()
            else:
                pred_values = predictions
            
            # Regression metrics
            mae = metrics.mean_absolute_error(labels, pred_values)
            mse = metrics.mean_squared_error(labels, pred_values)
            rmse = np.sqrt(mse)
            
            # R-squared score
            r2 = metrics.r2_score(labels, pred_values)
            
            # Mean Absolute Percentage Error (MAPE) with safe division
            mask = labels != 0
            if np.any(mask):
                mape = np.mean(np.abs((labels[mask] - pred_values[mask]) / labels[mask])) * 100
            else:
                mape = float('inf')
            
            # Pearson correlation
            try:
                pearson_r, _ = np.corrcoef(labels, pred_values)[0, 1], None
            except:
                pearson_r = 0.0
            
            result_metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape if mape != float('inf') else 0.0,
                'pearson_r': pearson_r,
            }
            
        else:  # classification
            pred_labels = np.argmax(predictions, axis=1)
            
            # Basic classification metrics
            accuracy = metrics.accuracy_score(labels, pred_labels)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                labels, pred_labels, average='weighted', zero_division=0
            )
            
            result_metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }
            
            # Cohen's kappa
            kappa = metrics.cohen_kappa_score(labels, pred_labels)
            result_metrics['kappa'] = kappa
            
            # MAE and RMSE for ordinal classification
            mae = metrics.mean_absolute_error(labels, pred_labels)
            rmse = np.sqrt(metrics.mean_squared_error(labels, pred_labels))
            result_metrics['mae'] = mae
            result_metrics['rmse'] = rmse
            
            # Multi-class specific metrics
            if self.config.num_labels > 2:
                precision_macro, recall_macro, f1_macro, _ = metrics.precision_recall_fscore_support(
                    labels, pred_labels, average='macro', zero_division=0
                )
                result_metrics.update({
                    'f1_macro': f1_macro,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                })
                
                try:
                    qwk = metrics.cohen_kappa_score(labels, pred_labels, weights='quadratic')
                    result_metrics['quadratic_weighted_kappa'] = qwk
                except (ValueError, RuntimeError) as e:
                    self.logger.warning(f"Could not compute quadratic weighted kappa: {e}")
            else:
                # Binary classification metrics
                try:
                    probs = predictions
                    if probs.shape[1] == 2:
                        auc = metrics.roc_auc_score(labels, probs[:, 1])
                        result_metrics['auc'] = auc
                except (ValueError, RuntimeError) as e:
                    self.logger.warning(f"Could not compute AUC: {e}")
        
        return result_metrics
    
    def _detect_overfitting(self, train_metrics: dict, val_metrics: dict):
        """Detect overfitting by comparing training and validation metrics"""
        if not self.config.enable_overfitting_detection:
            return False
        
        # Get the metric to monitor
        metric_name = self.config.overfitting_metric.lower()
        
        # Handle different metric names
        if metric_name == "loss":
            train_value = train_metrics.get('loss', train_metrics.get('train_loss', None))
            val_value = val_metrics.get('eval_loss', val_metrics.get('val_loss', None))
            lower_is_better = True
        elif metric_name == "accuracy":
            train_value = train_metrics.get('accuracy', train_metrics.get('train_accuracy', None))
            val_value = val_metrics.get('eval_accuracy', val_metrics.get('val_accuracy', None))
            lower_is_better = False
        elif metric_name == "f1":
            train_value = train_metrics.get('f1', train_metrics.get('train_f1', None))
            val_value = val_metrics.get('eval_f1', val_metrics.get('val_f1', None))
            lower_is_better = False
        else:
            self.logger.warning(f"Unknown overfitting metric: {metric_name}")
            return False
        
        if train_value is None or val_value is None:
            return False
        
        # Calculate absolute and relative differences
        abs_diff = abs(train_value - val_value)
        rel_diff = abs_diff / (abs(val_value) + 1e-8)  # Avoid division by zero
        
        # Check if overfitting based on metric direction
        is_overfit = False
        if lower_is_better:
            # For loss: training loss should not be much lower than validation loss
            if train_value < val_value:
                is_overfit = (abs_diff > self.config.overfitting_threshold or 
                            rel_diff > self.config.overfitting_relative_threshold)
        else:
            # For accuracy/f1: training metric should not be much higher than validation
            if train_value > val_value:
                is_overfit = (abs_diff > self.config.overfitting_threshold or 
                            rel_diff > self.config.overfitting_relative_threshold)
        
        # Update overfitting counter
        if is_overfit:
            self.overfitting_counter += 1
            if self.overfitting_counter >= self.config.overfitting_patience:
                self.is_overfitting = True
                self._handle_overfitting(train_value, val_value, metric_name)
        else:
            self.overfitting_counter = 0
            self.is_overfitting = False
        
        # Store metrics history
        self.train_metrics_history.append(train_value)
        self.val_metrics_history.append(val_value)
        
        return self.is_overfitting
    
    def _handle_overfitting(self, train_value: float, val_value: float, metric_name: str):
        """Handle detected overfitting based on configured action"""
        gap = abs(train_value - val_value)
        rel_gap = gap / (abs(val_value) + 1e-8) * 100
        
        self.logger.warning(f"⚠️  OVERFITTING DETECTED!")
        self.logger.warning(f"   Metric: {metric_name}")
        self.logger.warning(f"   Training: {train_value:.4f}")
        self.logger.warning(f"   Validation: {val_value:.4f}")
        self.logger.warning(f"   Gap: {gap:.4f} ({rel_gap:.1f}%)")
        
        if self.config.overfitting_action == "stop":
            self.logger.error("Stopping training due to overfitting")
            self.trainer.args.should_save = True
            self.trainer.control.should_training_stop = True
        elif self.config.overfitting_action == "reduce_lr":
            self.logger.info("Reducing learning rate due to overfitting")
            # Reduce learning rate by 50%
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] *= 0.5
            self.logger.info(f"New learning rate: {self.trainer.optimizer.param_groups[0]['lr']:.2e}")
            self.overfitting_counter = 0  # Reset counter after action
        else:  # "warn"
            self.logger.warning("Consider reducing learning rate or adding regularization")
    
    def setup_trainer(self):
        """Comprehensive trainer setup with all features"""
        transformers = get_module('transformers')
        torch = get_module('torch')
        
        self.logger.info("Initializing HuggingFace Trainer with advanced configurations")
        
        try:
            optimizer, scheduler = self._setup_optimizer_and_scheduler()
            
            # Compute class weights (only for classification)
            class_weights = None
            if self.config.task_type == "classification":
                if self.config.auto_class_weights and hasattr(self, '_original_train_dataset'):
                    class_weights = self._compute_class_weights(self._original_train_dataset)
                elif self.config.class_weights:
                    class_weights = torch.tensor(self.config.class_weights, dtype=torch.float32)
            
            if class_weights is not None:
                try:
                    if hasattr(self.model, 'device') and self.model.device is not None:
                        device = self.model.device
                    else:
                        # Safe way to get device from model parameters
                        try:
                            device = next(self.model.parameters()).device
                        except StopIteration:
                            device = self.device  # Fallback to configured device
                    class_weights = class_weights.to(device)
                    self.logger.info(f"Class weights transferred to device: {device}")
                except Exception as e:
                    self.logger.warning(f"Failed to transfer class weights to device: {e}")
            
            # Calculate total steps for progress bar
            if self.config.max_steps > 0:
                total_steps = self.config.max_steps
            else:
                steps_per_epoch = len(self.datasets['train']) // (
                    self.config.train_batch_size * self.config.gradient_accumulation_steps
                )
                total_steps = steps_per_epoch * self.config.max_epochs
            
            self._setup_progress_bar(total_steps, "Training")
            
            # Training arguments
            training_args = transformers.TrainingArguments(
                output_dir=self.config.output_dir,
                overwrite_output_dir=self.config.overwrite_output_dir,
                
                num_train_epochs=self.config.max_epochs,
                max_steps=self.config.max_steps,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                warmup_steps=self.config.warmup_steps,
                max_grad_norm=self.config.max_grad_norm,
                
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                tf32=self.config.tf32,
                fp16_full_eval=self.config.fp16,
                dataloader_drop_last=False,
                
                label_smoothing_factor=self.config.label_smoothing,
                
                gradient_checkpointing=self.config.gradient_checkpointing,
                dataloader_num_workers=self.config.dataloader_num_workers,
                dataloader_pin_memory=self.config.dataloader_pin_memory,
                dataloader_prefetch_factor=self.config.dataloader_prefetch_factor,
                
                eval_strategy=self.config.eval_strategy,
                eval_steps=self.config.eval_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                
                logging_steps=self.config.logging_steps,
                log_level=self.config.log_level,
                report_to=self.config.report_to,
                run_name=self.config.run_name,
                
                local_rank=self.config.local_rank,
                
                push_to_hub=self.config.push_to_hub,
                hub_model_id=self.config.hub_model_id,
                hub_strategy=self.config.hub_strategy,
                hub_token=self.config.hub_token,
                
                resume_from_checkpoint=self.config.resume_from_checkpoint,
                save_safetensors=self.config.save_safetensors,
                
                disable_tqdm=True,
                logging_first_step=False,
                logging_strategy="no",
                ignore_data_skip=True,
            )
            
            # Callbacks
            callbacks = [TrainingProgressCallback(self.progress_bar, self.logger)]
            
            # Add overfitting detection callback if enabled
            if self.config.enable_overfitting_detection:
                callbacks.append(OverfittingDetectionCallback(self))
            
            if self.config.early_stopping_patience > 0:
                callbacks.append(
                    transformers.EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience,
                        early_stopping_threshold=self.config.early_stopping_threshold,
                    )
                )
                self.logger.info(f"Early stopping configured with patience: {self.config.early_stopping_patience}")
            
            # Custom trainer with weighted loss
            class WeightedTrainer(transformers.Trainer):
                def __init__(self, class_weights=None, task_type="classification", **kwargs):
                    super().__init__(**kwargs)
                    self.class_weights = class_weights
                    self.task_type = task_type
                
                def compute_loss(self, model, inputs, return_outputs=False, **_unused_kwargs):
                    """
                    Custom loss computation with class weights support for classification and MSE for regression
                    Fixed signature to handle additional kwargs from newer transformers versions
                    """
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    
                    if self.task_type == "regression":
                        # For regression, use MSE loss
                        loss = outputs.loss  # Model already uses MSE for regression
                    elif labels is not None and self.class_weights is not None:
                        # Classification with class weights
                        torch = get_module('torch')
                        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                        # Handle DataParallel wrapper
                        model_config = model.module.config if isinstance(model, torch.nn.DataParallel) else model.config
                        loss = loss_fct(outputs.logits.view(-1, model_config.num_labels), 
                                      labels.view(-1))
                    else:
                        # Default loss from model
                        loss = outputs.loss
                    
                    return (loss, outputs) if return_outputs else loss
            
            # Initialize data collator
            data_collator = transformers.DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt"
            )
            
            # Initialize trainer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.trainer = WeightedTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.datasets.get('train'),
                    eval_dataset=self.datasets.get('validation'),
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    compute_metrics=self._compute_metrics,
                    optimizers=(optimizer, scheduler),
                    callbacks=callbacks,
                    class_weights=class_weights,
                    task_type=self.config.task_type,
                )
            
            self.logger.success("Trainer initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Trainer setup failed: {str(e)}")
            raise
    
    def train(self):
        """Execute comprehensive training pipeline"""
        global _global_trainer
        self.logger.info("Launching comprehensive training pipeline")
        
        # Set global trainer for signal handling
        _global_trainer = self
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.config.resume_from_checkpoint
            )
            
            if self.progress_bar:
                self.progress_bar.close()
                self.progress_bar = None
            
            self.logger.success("Training pipeline completed successfully")
            self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            if hasattr(train_result, 'metrics'):
                self.logger.info("Final training metrics summary:")
                for key, value in train_result.metrics.items():
                    if isinstance(value, float):
                        self.logger.info(f"  {key}: {value:.4f}")
                    else:
                        self.logger.info(f"  {key}: {value}")
            
            return train_result
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user - saving best model checkpoint")
            self._save_best_checkpoint_on_interrupt()
            raise
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            # Clean up global trainer reference
            _global_trainer = None
            
            if hasattr(self, 'epoch_progress_bar') and self.epoch_progress_bar:
                self.epoch_progress_bar.close()
                self.epoch_progress_bar = None
            if self.progress_bar:
                self.progress_bar.close()
                self.progress_bar = None
    
    def evaluate(self, dataset_name: str = "validation"):
        """Comprehensive evaluation with detailed metrics"""
        self.logger.info(f"Executing comprehensive evaluation on {dataset_name} dataset")
        
        if dataset_name not in self.datasets or self.datasets[dataset_name] is None:
            raise ValueError(f"Dataset '{dataset_name}' not available for evaluation")
        
        try:
            eval_results = self.trainer.evaluate(eval_dataset=self.datasets[dataset_name])
            
            self.logger.success(f"Evaluation completed successfully on {dataset_name} dataset")
            
            # Log results in organized format based on task type
            if self.config.task_type == "regression":
                # Regression metrics
                regression_metrics = ['eval_loss', 'eval_mae', 'eval_mse', 'eval_rmse', 'eval_r2', 'eval_mape', 'eval_pearson_r']
                
                self.logger.info("Regression Performance Metrics:")
                for key in regression_metrics:
                    if key in eval_results:
                        value = eval_results[key]
                        if isinstance(value, float):
                            metric_name = key.replace('eval_', '').upper()
                            if metric_name == 'R2':
                                metric_name = 'R²'
                            elif metric_name == 'PEARSON_R':
                                metric_name = 'Pearson r'
                            self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                # Classification metrics
                core_metrics = ['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']
                additional_metrics = ['eval_kappa', 'eval_mae', 'eval_rmse', 'eval_auc']
                advanced_metrics = ['eval_f1_macro', 'eval_precision_macro', 'eval_recall_macro', 
                                  'eval_quadratic_weighted_kappa']
                
                self.logger.info("Core Performance Metrics:")
                for key in core_metrics:
                    if key in eval_results:
                        value = eval_results[key]
                        if isinstance(value, float):
                            self.logger.info(f"  {key.replace('eval_', '').upper()}: {value:.4f}")
                            
                if any(key in eval_results for key in additional_metrics):
                    self.logger.info("Additional Performance Metrics:")
                    for key in additional_metrics:
                        if key in eval_results:
                            value = eval_results[key]
                            if isinstance(value, float):
                                self.logger.info(f"  {key.replace('eval_', '').upper()}: {value:.4f}")
                                
                if any(key in eval_results for key in advanced_metrics):
                    self.logger.info("Advanced Performance Metrics:")
                    for key in advanced_metrics:
                        if key in eval_results:
                            value = eval_results[key]
                            if isinstance(value, float):
                                self.logger.info(f"  {key.replace('eval_', '').upper()}: {value:.4f}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def predict(self, dataset_name: str = "test"):
        """Generate predictions with comprehensive output"""
        np = get_module('numpy')
        
        self.logger.info(f"Generating predictions on {dataset_name} dataset")
        
        if dataset_name not in self.datasets or self.datasets[dataset_name] is None:
            raise ValueError(f"Dataset '{dataset_name}' not available for prediction")
        
        try:
            predictions = self.trainer.predict(test_dataset=self.datasets[dataset_name])
            
            output_file = Path(self.config.output_dir) / "predictions" / f"{dataset_name}_predictions.npy"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, predictions.predictions)
            
            self.logger.success(f"Predictions generated and saved to {output_file}")
            
            return predictions.predictions
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {str(e)}")
            raise
    
    def save_model(self, save_path: Optional[str] = None):
        """Comprehensive model and artifact saving"""
        if save_path is None:
            save_path = Path(self.config.output_dir) / "final_model"
        else:
            save_path = Path(save_path)
        
        self.logger.info(f"Saving complete model state to {save_path}")
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            self.trainer.save_model(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            
            # Save training configuration
            config_file = save_path / "training_config.json"
            with open(config_file, 'w') as f:
                config_dict = {}
                for field_name, field_value in self.config.__dict__.items():
                    if isinstance(field_value, (list, dict, str, int, float, bool)) or field_value is None:
                        config_dict[field_name] = field_value
                    else:
                        config_dict[field_name] = str(field_value)
                json.dump(config_dict, f, indent=2)
            
            # Save trainer state
            if hasattr(self.trainer, 'state'):
                state_file = save_path / "trainer_state.json"
                self.trainer.state.save_to_json(str(state_file))
            
            self.logger.success(f"Model and configuration saved successfully to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save emergency checkpoint"""
        checkpoint_path = Path(self.config.output_dir) / "checkpoints" / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.save_model(str(checkpoint_path))
        self.logger.info(f"Emergency checkpoint saved to {checkpoint_path}")
    
    def _save_best_checkpoint_on_interrupt(self):
        """Save the best model checkpoint when training is interrupted"""
        try:
            # Check if we have a trainer with state
            if hasattr(self, 'trainer') and self.trainer is not None:
                
                # Try to find the best checkpoint manually by looking at saved checkpoints
                output_path = Path(self.config.output_dir)
                checkpoint_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
                
                best_checkpoint = None
                if hasattr(self.trainer, 'state') and self.trainer.state is not None:
                    # First try to get it from trainer state
                    best_checkpoint = self.trainer.state.best_model_checkpoint
                    
                    # If that's null but we have best_global_step, find the corresponding checkpoint
                    if not best_checkpoint and hasattr(self.trainer.state, 'best_global_step'):
                        best_step = self.trainer.state.best_global_step
                        if best_step:
                            best_checkpoint_dir = output_path / f"checkpoint-{best_step}"
                            if best_checkpoint_dir.exists():
                                best_checkpoint = str(best_checkpoint_dir)
                                self.logger.info(f"Found best checkpoint by step number: {best_checkpoint}")
                
                # If we found a best checkpoint, copy it
                if best_checkpoint and Path(best_checkpoint).exists():
                    interrupted_path = Path(self.config.output_dir) / "checkpoints" / "interrupted_best_model" 
                    interrupted_path.mkdir(parents=True, exist_ok=True)
                    
                    # Copy model files from best checkpoint
                    import shutil
                    best_checkpoint_path = Path(best_checkpoint)
                    
                    # Copy all files from best checkpoint
                    for file_path in best_checkpoint_path.iterdir():
                        if file_path.is_file():
                            shutil.copy2(file_path, interrupted_path / file_path.name)
                    
                    self.logger.success(f"Best model checkpoint saved to {interrupted_path}")
                    self.logger.info(f"Best model was from: {best_checkpoint}")
                    
                    # Also log the best metric for reference
                    if hasattr(self.trainer.state, 'best_metric'):
                        self.logger.info(f"Best metric ({self.config.metric_for_best_model}): {self.trainer.state.best_metric}")
                    return
                
                # Try to find the checkpoint with the best metric by examining all checkpoints
                elif checkpoint_dirs:
                    self.logger.info("Searching through saved checkpoints for best model...")
                    best_metric = None
                    best_checkpoint_path = None
                    
                    for checkpoint_dir in checkpoint_dirs:
                        trainer_state_file = checkpoint_dir / "trainer_state.json"
                        if trainer_state_file.exists():
                            try:
                                import json
                                with open(trainer_state_file, 'r') as f:
                                    state = json.load(f)
                                
                                # Look through log history for best metric
                                for log_entry in state.get('log_history', []):
                                    metric_key = self.config.metric_for_best_model
                                    if metric_key in log_entry:
                                        metric_value = log_entry[metric_key]
                                        if best_metric is None:
                                            best_metric = metric_value
                                            best_checkpoint_path = checkpoint_dir
                                        elif self.config.greater_is_better and metric_value > best_metric:
                                            best_metric = metric_value 
                                            best_checkpoint_path = checkpoint_dir
                                        elif not self.config.greater_is_better and metric_value < best_metric:
                                            best_metric = metric_value
                                            best_checkpoint_path = checkpoint_dir
                                            
                            except Exception as e:
                                self.logger.warning(f"Could not read trainer state from {checkpoint_dir}: {e}")
                    
                    if best_checkpoint_path:
                        interrupted_path = Path(self.config.output_dir) / "checkpoints" / "interrupted_best_model"
                        interrupted_path.mkdir(parents=True, exist_ok=True)
                        
                        import shutil
                        for file_path in best_checkpoint_path.iterdir():
                            if file_path.is_file():
                                shutil.copy2(file_path, interrupted_path / file_path.name)
                        
                        self.logger.success(f"Best model checkpoint saved to {interrupted_path}")
                        self.logger.info(f"Best model was from: {best_checkpoint_path}")
                        self.logger.info(f"Best metric ({self.config.metric_for_best_model}): {best_metric}")
                        return
                
                # Final fallback: save current model state
                self.logger.warning("Could not find best checkpoint, saving current model state")
                self._save_checkpoint("interrupted_current")
            else:
                self.logger.error("No trainer available to save checkpoint")
                
        except Exception as e:
            self.logger.error(f"Failed to save best checkpoint: {str(e)}")
            # Fallback to regular checkpoint saving
            try:
                self._save_checkpoint("interrupted_fallback")
            except Exception as fallback_error:
                self.logger.error(f"Fallback checkpoint save also failed: {str(fallback_error)}")
    
    @classmethod
    def load_model(cls, model_path: str, logger: logging.Logger):
        """Load pre-trained model with full configuration"""
        transformers = get_module('transformers')
        
        model_path = Path(model_path)
        logger.info(f"Loading pre-trained model from {model_path}")
        
        try:
            config_file = model_path / "training_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                config = TrainingConfig(**config_dict)
            else:
                logger.warning("Training config not found - using default configuration")
                config = TrainingConfig()
            
            trainer = cls(config, logger)
            
            trainer.tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
            trainer.model = transformers.AutoModelForSequenceClassification.from_pretrained(str(model_path))
            trainer.model.to(trainer.device)
            
            logger.success(f"Model loaded successfully from {model_path}")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

class OverfittingDetectionCallback(transformers.TrainerCallback):
    """Callback for detecting and handling overfitting during training"""
    
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance
        self.train_loss = None
        self.train_metrics = {}
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track training metrics"""
        if logs:
            # Update training metrics
            if 'loss' in logs:
                self.train_loss = logs['loss']
                self.train_metrics['loss'] = logs['loss']
            
            # Check for evaluation metrics
            if any(key.startswith('eval_') for key in logs.keys()):
                # We have validation metrics, check for overfitting
                if self.train_loss is not None:
                    train_metrics = {'loss': self.train_loss}
                    train_metrics.update(self.train_metrics)
                    
                    # Call overfitting detection
                    is_overfitting = self.trainer_instance._detect_overfitting(
                        train_metrics, logs
                    )
                    
                    # Update control if needed
                    if is_overfitting and self.trainer_instance.config.overfitting_action == "stop":
                        control.should_training_stop = True
        
        return control
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Additional check after evaluation"""
        if metrics and self.train_loss is not None:
            train_metrics = {'loss': self.train_loss}
            train_metrics.update(self.train_metrics)
            
            is_overfitting = self.trainer_instance._detect_overfitting(
                train_metrics, metrics
            )
            
            if is_overfitting and self.trainer_instance.config.overfitting_action == "stop":
                control.should_training_stop = True
        
        return control

class TrainingProgressCallback(transformers.TrainerCallback):
    """Comprehensive training progress callback with per-epoch progress bars"""
    
    def __init__(self, progress_bar, logger: logging.Logger):
        self.progress_bar = progress_bar
        self.logger = logger
        self.current_epoch = 0
        self.last_logged_step = 0
        self.epoch_progress_bar = None
        self.steps_per_epoch = 0
        
    def on_train_begin(self, _args, _state, _control, **_kwargs):
        if self.progress_bar:
            self.progress_bar.reset()
        self.logger.info("Training started")
    
    def on_epoch_begin(self, _args, state, _control, **_kwargs):
        self.current_epoch = int(state.epoch) if state.epoch else 0
        
        # Calculate steps per epoch
        if hasattr(state, 'max_steps') and hasattr(state, 'num_train_epochs'):
            self.steps_per_epoch = state.max_steps // state.num_train_epochs
        else:
            self.steps_per_epoch = 100  # fallback
        
        # Close previous epoch progress bar if exists
        if self.epoch_progress_bar:
            self.epoch_progress_bar.close()
            self.epoch_progress_bar = None
        
        # Skip creating epoch progress bar - we already have the main training progress bar
        
        if self.progress_bar:
            self.progress_bar.set_description(f"Epoch {self.current_epoch + 1}")
        self.logger.info(f"Starting epoch {self.current_epoch + 1}")
    
    def on_log(self, _args, state, _control, logs=None, **_kwargs):
        if logs:
            # Check if this is evaluation metrics
            is_eval_metrics = any(key.startswith('eval_') for key in logs.keys())
            
            # Update main progress bar without metrics to keep it clean
            if self.progress_bar:
                # Update progress
                if hasattr(state, 'global_step'):
                    self.progress_bar.n = state.global_step
                    self.progress_bar.refresh()
            
            # Update epoch progress bar for training steps
            if self.epoch_progress_bar and not is_eval_metrics:
                epoch_step = state.global_step % self.steps_per_epoch if self.steps_per_epoch > 0 else 0
                self.epoch_progress_bar.n = epoch_step
                self.epoch_progress_bar.refresh()
                
            # Handle evaluation metrics output
            if is_eval_metrics:
                # Ensure progress bar stays at bottom after evaluation output
                if self.progress_bar:
                    # Force a complete refresh to reposition the progress bar
                    self.progress_bar.clear()
                    self.progress_bar.refresh()
    
    def on_step_end(self, _args, _state, _control, **_kwargs):
        if self.progress_bar:
            self.progress_bar.update(1)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Ensure progress bar is properly positioned after evaluation"""
        if self.progress_bar:
            # Force a complete refresh after evaluation metrics are displayed
            self.progress_bar.clear()
            self.progress_bar.refresh()
        return control
    
    def on_train_end(self, _args, _state, _control, **_kwargs):
        if self.epoch_progress_bar:
            self.epoch_progress_bar.close()
            self.epoch_progress_bar = None
        if self.progress_bar:
            self.progress_bar.close()
        self.logger.success("Training completed")

def create_argument_parser():
    """Create comprehensive argument parser with rich formatting"""
    import argparse
    
    try:
        import rich_argparse
        formatter_class = rich_argparse.RichHelpFormatter
    except ImportError:
        formatter_class = argparse.HelpFormatter
    
    parser = argparse.ArgumentParser(
        description="Advanced BERT Training Pipeline with State-of-the-Art Features and Comprehensive Logging",
        formatter_class=formatter_class
    )
    
    # Data input group
    data_group = parser.add_argument_group("Data Input Configuration")
    data_input = data_group.add_mutually_exclusive_group(required=True)
    data_input.add_argument(
        "-i", "--input", 
        type=str,
        help="Single input file (xlsx/csv/parquet/json/tsv) for automatic train/val/test splitting (default: None)"
    )
    data_input.add_argument(
        "--train-set",
        type=str,
        help="Pre-split training dataset file - requires --val-set for validation (default: None)"
    )
    
    parser.add_argument(
        "--val-set",
        type=str,
        help="Pre-split validation dataset file - used for model selection and early stopping (default: None)"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        help="Pre-split test dataset file - used for final model evaluation (default: None)"
    )
    
    # Data processing configuration
    data_config = parser.add_argument_group("Data Processing Configuration")
    data_config.add_argument(
        "--text-col", "--text-column",
        type=str,
        default="text",
        help="Name of text column in dataset - must match exact column header (default: text)"
    )
    data_config.add_argument(
        "--label-col", "--label-column",
        type=str,
        default="label",
        help="Name of label column in dataset - must contain integer labels starting from 0 (default: label)"
    )
    data_config.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio for --input mode, range 0.05-0.4, applied after test split (default: 0.2)"
    )
    data_config.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split ratio for --input mode, range 0.0-0.3, applied first before validation (default: 0.1)"
    )
    data_config.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting (stratification is enabled by default to maintain label distribution)"
    )
    data_config.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length, range 64-8192, longer sequences require more memory (default: 512)"
    )
    
    # Model architecture group
    model_group = parser.add_argument_group("Model Architecture Configuration")
    model_group.add_argument(
        "--model", "--model-name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="HuggingFace model identifier - supports BERT, RoBERTa, DeBERTa, ModernBERT variants (default: answerdotai/ModernBERT-base)"
    )
    model_group.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type: classification for discrete labels, regression for continuous targets (default: classification)"
    )
    model_group.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of classification classes, range 2-1000, auto-detected from training data (default: 2)"
    )
    model_group.add_argument(
        "--custom-head",
        type=int,
        nargs="*",
        help="Custom classifier layer sizes, example: 768 256 for two-layer head (default: None)"
    )
    model_group.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["cls", "mean", "max", "attention"],
        help="Token embedding aggregation strategy for classification (default: cls)"
    )
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Initial learning rate, range 1e-6 to 1e-4, lower for large models, higher for small datasets (default: 2e-5)"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Maximum training epochs, range 1-100, use early stopping to prevent overfitting (default: 3)"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size per device, range 1-128, adjust based on GPU memory availability (default: 16)"
    )
    train_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Evaluation batch size per device, range 8-256, can be larger than training batch (default: 32)"
    )
    train_group.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps, range 1-32, simulates larger batch size when memory limited (default: 1)"
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="L2 regularization strength, range 0.0-0.3, prevents overfitting by penalizing large weights (default: 0.01)"
    )
    train_group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Linear warmup proportion of total training, range 0.0-0.3, stabilizes early training (default: 0.1)"
    )
    train_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold, range 0.1-10.0, prevents exploding gradients (default: 1.0)"
    )
    train_group.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor, range 0.0-0.3, reduces overconfidence and improves calibration (default: 0.0)"
    )
    
    # Optimization strategy
    opt_group = parser.add_argument_group("Optimization Strategy")
    opt_group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimization algorithm - AdamW recommended for transformers, SGD requires lower learning rate (default: adamw)"
    )
    opt_group.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_restarts", "polynomial", "constant"],
        help="Learning rate scheduling strategy - linear standard for BERT, cosine for longer training (default: linear)"
    )
    opt_group.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Fixed warmup steps, range 0-5000, overrides --warmup-ratio when specified (default: 0)"
    )
    
    # Transfer learning configuration
    transfer_group = parser.add_argument_group("Transfer Learning Configuration")
    transfer_group.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding layers during training - useful for domain adaptation (default: False)"
    )
    transfer_group.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Number of bottom transformer layers to freeze, range 0-12, preserves low-level features (default: 0)"
    )
    transfer_group.add_argument(
        "--progressive-unfreezing",
        action="store_true",
        help="Gradually unfreeze layers during training - prevents catastrophic forgetting (default: False)"
    )
    transfer_group.add_argument(
        "--layer-lr-decay",
        type=float,
        default=1.0,
        help="Layer-wise learning rate decay factor, range 0.1-1.0, lower layers get reduced learning rates (default: 1.0)"
    )
    transfer_group.add_argument(
        "--unfreeze-schedule",
        type=int,
        nargs="*",
        help="Specific epochs for layer unfreezing with --progressive-unfreezing (default: None)"
    )
    
    # Performance optimization
    perf_group = parser.add_argument_group("Performance Optimization")
    perf_group.add_argument(
        "--fp16",
        action="store_true",
        help="Enable 16-bit mixed precision training - reduces memory usage and increases speed (default: False)"
    )
    perf_group.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BFloat16 precision - better stability than FP16, requires Ampere+ GPUs (default: False)"
    )
    perf_group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade computation for memory - enables larger models/batches at cost of training speed (default: False)"
    )
    
    # Evaluation and checkpointing
    eval_group = parser.add_argument_group("Evaluation and Checkpointing")
    eval_group.add_argument(
        "--eval-strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Evaluation timing strategy - steps for large datasets, epoch for small datasets (default: steps)"
    )
    eval_group.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation frequency in training steps, range 50-2000, balance monitoring with training speed (default: 500)"
    )
    eval_group.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint saving frequency in steps, range 100-5000, enables training recovery (default: 500)"
    )
    eval_group.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="Early stopping patience in evaluations, range 1-10, prevents overfitting (default: 5)"
    )
    eval_group.add_argument(
        "--metric-for-best",
        type=str,
        default="eval_loss",
        help="Metric for best model selection - eval_loss, eval_accuracy, eval_f1, choose task-appropriate metric (default: eval_loss)"
    )
    eval_group.add_argument(
        "--greater-is-better",
        action="store_true",
        help="Whether higher metric values indicate better performance - True for accuracy/F1, False for loss (default: False)"
    )
    eval_group.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum checkpoints to retain, range 1-10, older checkpoints automatically deleted (default: 3)"
    )
    eval_group.add_argument(
        "--load-best-at-end",
        action="store_true",
        help="Load best validation checkpoint at training completion - recommended for final model (default: False)"
    )
    
    # Overfitting detection
    overfit_group = parser.add_argument_group("Overfitting Detection and Prevention")
    overfit_group.add_argument(
        "--disable-overfitting-detection",
        action="store_true",
        help="Disable automatic overfitting detection (default: enabled)"
    )
    overfit_group.add_argument(
        "--overfitting-threshold",
        type=float,
        default=0.05,
        help="Absolute gap threshold between train/val metrics for overfitting detection (default: 0.05)"
    )
    overfit_group.add_argument(
        "--overfitting-relative-threshold",
        type=float,
        default=0.1,
        help="Relative gap threshold (percentage) for overfitting detection (default: 0.1 = 10%%)"
    )
    overfit_group.add_argument(
        "--overfitting-patience",
        type=int,
        default=3,
        help="Number of evaluations with overfitting before taking action (default: 3)"
    )
    overfit_group.add_argument(
        "--overfitting-metric",
        type=str,
        default="loss",
        choices=["loss", "accuracy", "f1"],
        help="Metric to monitor for overfitting detection (default: loss)"
    )
    overfit_group.add_argument(
        "--overfitting-action",
        type=str,
        default="warn",
        choices=["warn", "stop", "reduce_lr"],
        help="Action to take when overfitting is detected - warn/stop/reduce_lr (default: warn)"
    )
    
    # Loss function configuration
    loss_group = parser.add_argument_group("Loss Function Configuration")
    loss_group.add_argument(
        "--loss-type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal"],
        help="Loss function type - focal loss addresses class imbalance by focusing on hard examples (default: cross_entropy)"
    )
    loss_group.add_argument(
        "--class-weights",
        type=float,
        nargs="*",
        help="Manual per-class weights for imbalanced datasets, must match number of classes (default: None)"
    )
    loss_group.add_argument(
        "--auto-class-weights",
        action="store_true",
        help="Automatically compute inverse frequency class weights for imbalanced data (default: False)"
    )
    loss_group.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss class balance parameter, range 0.0-1.0, controls minority/majority class weighting (default: 0.25)"
    )
    loss_group.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing parameter, range 0.0-5.0, higher values focus more on hard examples (default: 2.0)"
    )
    
    # Data resampling strategies
    resample_group = parser.add_argument_group("Data Resampling Strategies")
    resample_group.add_argument(
        "--resampling",
        type=str,
        default="none",
        choices=["none", "oversample", "undersample", "smote"],
        help="Training data resampling strategy for handling class imbalance (default: none)"
    )
    resample_group.add_argument(
        "--oversample-ratio",
        type=float,
        default=1.0,
        help="Target minority class ratio for oversampling, range 0.1-1.0 (default: 1.0)"
    )
    resample_group.add_argument(
        "--undersample-ratio",
        type=float,
        default=1.0,
        help="Majority class retention ratio for undersampling, range 0.1-1.0 (default: 1.0)"
    )
    
    # Data augmentation techniques
    aug_group = parser.add_argument_group("Data Augmentation Techniques")
    aug_group.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=["none", "basic"],
        help="Text augmentation strategy - basic includes synonym replacement, insertion, swap, deletion (default: none)"
    )
    aug_group.add_argument(
        "--aug-prob",
        type=float,
        default=0.5,
        help="Probability of applying augmentation to training samples, range 0.0-1.0 (default: 0.5)"
    )
    aug_group.add_argument(
        "--synonym-prob",
        type=float,
        default=0.1,
        help="Probability of synonym replacement augmentation, range 0.0-0.5 (default: 0.1)"
    )
    aug_group.add_argument(
        "--insert-prob",
        type=float,
        default=0.1,
        help="Probability of random word insertion augmentation, range 0.0-0.3 (default: 0.1)"
    )
    aug_group.add_argument(
        "--swap-prob",
        type=float,
        default=0.1,
        help="Probability of random word swap augmentation, range 0.0-0.3 (default: 0.1)"
    )
    aug_group.add_argument(
        "--delete-prob",
        type=float,
        default=0.1,
        help="Probability of random word deletion augmentation, range 0.0-0.3 (default: 0.1)"
    )
    
    # Hardware configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device selection - auto detects best available device (CUDA > MPS > CPU) (default: auto)"
    )
    hw_group.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Specific GPU device ID for multi-GPU systems, range 0 to num_gpus-1 (default: None)"
    )
    hw_group.add_argument(
        "--no-dataparallel",
        action="store_true",
        help="Disable DataParallel for multi-GPU systems - forces single GPU training (default: False)"
    )
    hw_group.add_argument(
        "--flash-attention",
        action="store_true",
        default=True,
        help="Enable Flash Attention 2.0 for memory and speed optimization on compatible GPUs (default: True)"
    )
    hw_group.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Explicitly disable Flash Attention 2.0 - use standard attention implementation (default: False)"
    )
    
    # Additional configuration
    additional_group = parser.add_argument_group("Additional Configuration")
    additional_group.add_argument(
        "-o", "--output", "--output-dir",
        type=str,
        default="./results",
        help="Output directory for models, checkpoints, logs, and predictions (default: ./results)"
    )
    additional_group.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Parallel data loading workers, range 0-16, optimize based on CPU cores available (default: 4)"
    )
    additional_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging with detailed progress information (default: False)"
    )
    additional_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with maximum verbosity for troubleshooting (default: False)"
    )
    additional_group.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite existing output directory - use with caution as it deletes existing results (default: False)"
    )
    additional_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training across Python, NumPy, PyTorch, and CUDA (default: 42)"
    )
    additional_group.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training - automatically set by launcher tools (default: -1)"
    )
    
    # Training resumption
    resume_group = parser.add_argument_group("Training Resumption")
    resume_group.add_argument(
        "--resume-from",
        type=str,
        help="Checkpoint path to resume training - restores exact training state including optimizer (default: None)"
    )
    resume_group.add_argument(
        "--load-model",
        type=str,
        help="Pre-trained model path to initialize training - different from resume, starts fresh training (default: None)"
    )
    resume_group.add_argument(
        "--run-name",
        type=str,
        help="Experiment identifier for output directory naming and logging systems (default: None)"
    )
    
    return parser

def main():
    """Main training pipeline with comprehensive error handling"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    logger.info("Initializing Advanced BERT Training Pipeline")
    
    if hasattr(args, 'seed'):
        torch = get_module('torch')
        np = get_module('numpy')
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed configured for reproducibility: {args.seed}")
    
    try:
        # Create comprehensive configuration
        config = TrainingConfig(
            # Model configuration
            model_name=args.model,
            task_type=args.task,
            num_labels=args.num_labels,
            max_position_embeddings=args.max_seq_length,
            
            # Custom head configuration
            custom_head_layers=args.custom_head if args.custom_head else [],
            pooling_strategy=args.pooling,
            
            # Training hyperparameters
            learning_rate=args.learning_rate,
            max_epochs=args.epochs,
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            label_smoothing=args.label_smoothing if hasattr(args, 'label_smoothing') else 0.0,
            
            # Optimization
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps if hasattr(args, 'warmup_steps') else 0,
            
            # Precision
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            
            # Transfer learning
            freeze_embeddings=args.freeze_embeddings if hasattr(args, 'freeze_embeddings') else False,
            freeze_encoder_layers=args.freeze_layers if hasattr(args, 'freeze_layers') else 0,
            progressive_unfreezing=args.progressive_unfreezing if hasattr(args, 'progressive_unfreezing') else False,
            layer_wise_lr_decay=args.layer_lr_decay if hasattr(args, 'layer_lr_decay') else 1.0,
            unfreeze_schedule=args.unfreeze_schedule if hasattr(args, 'unfreeze_schedule') else None,
            
            # Loss function
            loss_type=(
                "focal" if (hasattr(args, 'focal_alpha') and args.focal_alpha is not None) or
                          args.auto_class_weights
                else (args.loss_type if hasattr(args, 'loss_type') else "cross_entropy")
            ),
            class_weights=args.class_weights if hasattr(args, 'class_weights') else None,
            auto_class_weights=args.auto_class_weights if hasattr(args, 'auto_class_weights') else False,
            focal_loss_alpha=args.focal_alpha if hasattr(args, 'focal_alpha') else 0.25,
            focal_loss_gamma=args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0,
            
            # Data resampling
            resampling_strategy=args.resampling if hasattr(args, 'resampling') else "none",
            oversample_ratio=args.oversample_ratio if hasattr(args, 'oversample_ratio') else 1.0,
            undersample_ratio=args.undersample_ratio if hasattr(args, 'undersample_ratio') else 1.0,
            
            # Data augmentation
            augmentation_strategy=args.augmentation if hasattr(args, 'augmentation') else "none",
            augmentation_prob=args.aug_prob if hasattr(args, 'aug_prob') else 0.5,
            synonym_replacement_prob=args.synonym_prob if hasattr(args, 'synonym_prob') else 0.1,
            random_insertion_prob=args.insert_prob if hasattr(args, 'insert_prob') else 0.1,
            random_swap_prob=args.swap_prob if hasattr(args, 'swap_prob') else 0.1,
            random_deletion_prob=args.delete_prob if hasattr(args, 'delete_prob') else 0.1,
            
            # Evaluation
            eval_strategy=args.eval_strategy if hasattr(args, 'eval_strategy') else "steps",
            eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 500,
            save_steps=args.save_steps if hasattr(args, 'save_steps') else 500,
            early_stopping_patience=args.early_stopping if hasattr(args, 'early_stopping') else 5,
            metric_for_best_model=args.metric_for_best if hasattr(args, 'metric_for_best') else "eval_loss",
            greater_is_better=args.greater_is_better if hasattr(args, 'greater_is_better') else False,
            save_total_limit=args.save_total_limit if hasattr(args, 'save_total_limit') else 3,
            load_best_model_at_end=args.load_best_at_end if hasattr(args, 'load_best_at_end') else False,
            
            # Overfitting detection
            enable_overfitting_detection=not args.disable_overfitting_detection if hasattr(args, 'disable_overfitting_detection') else True,
            overfitting_threshold=args.overfitting_threshold if hasattr(args, 'overfitting_threshold') else 0.05,
            overfitting_relative_threshold=args.overfitting_relative_threshold if hasattr(args, 'overfitting_relative_threshold') else 0.1,
            overfitting_patience=args.overfitting_patience if hasattr(args, 'overfitting_patience') else 3,
            overfitting_metric=args.overfitting_metric if hasattr(args, 'overfitting_metric') else "loss",
            overfitting_action=args.overfitting_action if hasattr(args, 'overfitting_action') else "warn",
            
            # Data configuration
            text_column=args.text_col,
            label_column=args.label_col,
            validation_split_size=args.val_split,
            test_split_size=args.test_split,
            stratify_split=not args.no_stratify,
            max_seq_length=args.max_seq_length,
            
            # Hardware
            device=args.device if hasattr(args, 'device') else "auto",
            gpu_id=args.gpu_id if hasattr(args, 'gpu_id') else None,
            no_dataparallel=args.no_dataparallel if hasattr(args, 'no_dataparallel') else False,
            flash_attention=args.flash_attention if hasattr(args, 'flash_attention') else True,
            no_flash_attention=args.no_flash_attention if hasattr(args, 'no_flash_attention') else False,
            dataloader_num_workers=args.jobs,
            
            # Output
            output_dir=args.output,
            overwrite_output_dir=args.overwrite_output,
            resume_from_checkpoint=args.resume_from if hasattr(args, 'resume_from') else None,
            report_to=[],
            run_name=args.run_name if hasattr(args, 'run_name') else None,
            local_rank=args.local_rank,
        )
        
        # Initialize trainer
        if args.load_model:
            logger.info("Loading pre-trained model for continued training")
            trainer = AdvancedBERTTrainer.load_model(args.load_model, logger)
            # Override loaded config with CLI arguments
            trainer.config.output_dir = args.output
            if args.learning_rate != config.learning_rate:
                trainer.config.learning_rate = args.learning_rate
                logger.info(f"Overriding learning rate: {args.learning_rate}")
            if args.train_batch_size != config.train_batch_size:
                trainer.config.train_batch_size = args.train_batch_size
                logger.info(f"Overriding batch size: {args.train_batch_size}")
            if args.epochs != config.epochs:
                trainer.config.epochs = args.epochs
                logger.info(f"Overriding epochs: {args.epochs}")
            if args.warmup_steps != config.warmup_steps:
                trainer.config.warmup_steps = args.warmup_steps
                logger.info(f"Overriding warmup steps: {args.warmup_steps}")
        else:
            trainer = AdvancedBERTTrainer(config, logger)
            trainer.setup_tokenizer()
            
            # Load data
            if args.input:
                trainer.load_data(input_file=args.input)
            else:
                trainer.load_data(
                    train_file=args.train_set,
                    val_file=args.val_set,
                    test_file=args.test_set
                )
            
            trainer.setup_model()
        
        trainer.setup_trainer()
        
        # Execute training
        logger.info("Executing training pipeline")
        trainer.train()
        
        # Comprehensive evaluation
        final_metrics = {}
        
        if 'validation' in trainer.datasets and trainer.datasets['validation'] is not None:
            logger.info("Executing validation set evaluation")
            eval_results = trainer.evaluate('validation')
            final_metrics['validation'] = eval_results
        
        if 'test' in trainer.datasets and trainer.datasets['test'] is not None:
            logger.info("Executing test set evaluation")
            test_results = trainer.evaluate('test')
            final_metrics['test'] = test_results
            
            logger.info("Generating test set predictions")
            trainer.predict('test')
        
        # Save final model
        logger.info("Saving final model and training artifacts")
        trainer.save_model()
        
        # Final summary
        logger.info("TRAINING PIPELINE EXECUTION COMPLETED")
        logger.info(f"Model Architecture: {args.model}")
        logger.info(f"Dataset Source: {args.input or args.train_set}")
        logger.info(f"Training Samples Processed: {len(trainer.datasets.get('train', []))}")
        
        if 'validation' in final_metrics:
            val_acc = final_metrics['validation'].get('eval_accuracy', 'N/A')
            val_f1 = final_metrics['validation'].get('eval_f1', 'N/A')
            logger.info(f"Final Validation Accuracy: {val_acc:.4f}" if isinstance(val_acc, float) else f"Final Validation Accuracy: {val_acc}")
            logger.info(f"Final Validation F1 Score: {val_f1:.4f}" if isinstance(val_f1, float) else f"Final Validation F1 Score: {val_f1}")
            
        if 'test' in final_metrics:
            test_acc = final_metrics['test'].get('eval_accuracy', 'N/A')
            test_f1 = final_metrics['test'].get('eval_f1', 'N/A')
            logger.info(f"Final Test Accuracy: {test_acc:.4f}" if isinstance(test_acc, float) else f"Final Test Accuracy: {test_acc}")
            logger.info(f"Final Test F1 Score: {test_f1:.4f}" if isinstance(test_f1, float) else f"Final Test F1 Score: {test_f1}")
        
        logger.success("Training pipeline completed successfully")
        logger.info(f"All results and artifacts saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.warning("Training pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()