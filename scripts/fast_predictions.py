#!/usr/bin/env python3

import argparse
import json
import logging
import signal
import sys
import warnings
import tempfile
import shutil
import gc
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
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

def clear_gpu_memory():
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()

def disable_internal_compilation():
	try:
		import torch._dynamo
		torch._dynamo.config.suppress_errors = True
		torch._dynamo.config.disable = True
	except ImportError:
		pass

def get_file_info(input_path: Path) -> Tuple[Optional[int], List[str]]:
	suffix = input_path.suffix.lower()
	
	try:
		if suffix == '.csv':
			df_sample = pd.read_csv(input_path, nrows=1)
			with open(input_path, 'r') as f:
				total_rows = sum(1 for _ in f) - 1
		elif suffix == '.parquet':
			df_full = pd.read_parquet(input_path)
			total_rows = len(df_full)
			df_sample = df_full.head(1)
			del df_full
		elif suffix in ['.xlsx', '.xls']:
			df_full = pd.read_excel(input_path)
			total_rows = len(df_full)
			df_sample = df_full.head(1)
			del df_full
		else:
			raise ValueError(f"Unsupported file format: {suffix}")
		
		return total_rows, list(df_sample.columns)
	
	except Exception as e:
		logger.warning(f"Could not get file info: {e}")
		try:
			if suffix == '.csv':
				df_sample = pd.read_csv(input_path, nrows=1)
			elif suffix == '.parquet':
				df_sample = pd.read_parquet(input_path).head(1)
			else:
				df_sample = pd.read_excel(input_path, nrows=1)
			return None, list(df_sample.columns)
		except Exception:
			return None, []

def load_dataframe_chunked(input_path: Path, chunksize: Optional[int] = None, **kwargs) -> Iterator[pd.DataFrame]:
	suffix = input_path.suffix.lower()
	
	if chunksize is None:
		if suffix == '.csv':
			yield pd.read_csv(input_path, **kwargs)
		elif suffix in ['.xlsx', '.xls']:
			yield pd.read_excel(input_path, **kwargs)
		elif suffix == '.parquet':
			yield pd.read_parquet(input_path, **kwargs)
		else:
			raise ValueError(f"Unsupported file format: {suffix}")
		return
	
	if suffix == '.csv':
		for chunk in pd.read_csv(input_path, chunksize=chunksize, **kwargs):
			yield chunk
	elif suffix == '.parquet':
		parquet_file = pd.read_parquet(input_path, **kwargs)
		for i in range(0, len(parquet_file), chunksize):
			yield parquet_file.iloc[i:i+chunksize].copy()
		del parquet_file
	elif suffix in ['.xlsx', '.xls']:
		df_full = pd.read_excel(input_path, **kwargs)
		for i in range(0, len(df_full), chunksize):
			yield df_full.iloc[i:i+chunksize].copy()
		del df_full
	else:
		raise ValueError(f"Unsupported file format: {suffix}")

class OptimizedTextDataset(Dataset):
	def __init__(self, texts: List[str]):
		self.texts = texts
	
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		return self.texts[idx]

def collate_fn_optimized(batch, tokenizer, max_length: int):
	encoded = tokenizer(
		batch,
		truncation=True,
		padding='max_length',
		max_length=max_length,
		return_tensors='pt'
	)
	
	return {
		'input_ids': encoded['input_ids'],
		'attention_mask': encoded['attention_mask']
	}

def merge_prediction_files(temp_dir: Path, output_path: Path, output_format: str) -> None:
	chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
	if not chunk_files:
		raise ValueError("No chunk files found to merge")
	
	logger.info(f"Merging {len(chunk_files)} prediction chunks")
	
	merged_dfs = []
	for chunk_file in tqdm(chunk_files, desc="Merging chunks", colour='blue', leave=False):
		chunk_df = pd.read_parquet(chunk_file)
		merged_dfs.append(chunk_df)
	
	final_df = pd.concat(merged_dfs, ignore_index=True)
	
	if output_format == 'csv':
		final_df.to_csv(output_path, index=False)
	elif output_format == 'excel':
		final_df.to_excel(output_path, index=False)
	elif output_format == 'parquet':
		final_df.to_parquet(output_path, index=False)
	else:
		raise ValueError(f"Unsupported output format: {output_format}")
	
	logger.info(f"Merged predictions saved to {output_path}")

class BERTPredictor:
	def __init__(self, model_path: Path, use_fp16: bool = None):
		disable_internal_compilation()

		self.model_path = model_path
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if self.device == 'cpu':
			logger.warning("No CUDA device available, running on CPU")
		else:
			props = torch.cuda.get_device_properties(0)
			free_gb = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024 ** 3
			logger.info(f"Using {props.name} ({free_gb:.1f}GB free)")

		with open(self.model_path / 'config.json', 'r') as f:
			self.config = json.load(f)

		self.use_fp16 = use_fp16 if use_fp16 is not None else self.config.get('fp16', False)
		self.max_length = self.config.get('max_length', 512)

		self.tokenizer = self._load_tokenizer()
		self.model = self._load_model()
		logger.info(f"Loaded predictor on {self.device} (fp16: {self.use_fp16})")

	def _load_tokenizer(self):
		return AutoTokenizer.from_pretrained(self.model_path)
	
	def _load_model(self):
		bert_model_name = self.config['bert_model']
		num_labels = self.config['model_config']['num_labels']
		dropout = self.config['model_config'].get('dropout', 0.1)
		
		class SequenceClassifier(nn.Module):
			def __init__(self, model_name_or_path: str, num_labels: int, dropout: float = 0.1):
				super().__init__()
				self.num_labels = num_labels
				self.bert = AutoModelForSequenceClassification.from_pretrained(
					model_name_or_path, 
					num_labels=num_labels,
					ignore_mismatched_sizes=True
				)
			
			def forward(self, input_ids, attention_mask, labels=None, **kwargs):
				return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
		
		state_dict = torch.load(self.model_path / 'best_model.pt', map_location='cpu', weights_only=True)
		
		model = SequenceClassifier(bert_model_name, num_labels, dropout)
		model.load_state_dict(state_dict)
		model.eval()
		
		model = model.to(self.device)
		if self.device == 'cuda':
			torch.backends.cudnn.benchmark = True
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True

		return model
	
	def predict_chunk(self, texts: List[str], input_data: pd.DataFrame, batch_size: int = 128, 
					  return_probabilities: bool = False, num_workers: int = 4, 
					  label_mapping: Dict[int, str] = None) -> pd.DataFrame:
		
		dataset = OptimizedTextDataset(texts)
		# functools.partial, not a closure: DataLoader workers must pickle it
		# on spawn-based platforms (macOS, Windows).
		collate_fn = partial(collate_fn_optimized,
							 tokenizer=self.tokenizer, max_length=self.max_length)
		
		dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=(self.device == 'cuda'),
			collate_fn=collate_fn,
			drop_last=False
		)
		
		all_predictions = []
		all_probabilities = [] if return_probabilities else None
		
		context_manager = (torch.amp.autocast('cuda') if self.use_fp16 and self.device == 'cuda'
						   else torch.no_grad())
		
		with context_manager:
			for batch in dataloader:
				if killer.kill_now:
					break
				
				try:
					input_ids = batch['input_ids'].to(self.device, non_blocking=True)
					attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
					
					with torch.no_grad():
						outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
					
					logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
					
					predictions = torch.argmax(logits, dim=1).cpu().numpy()
					all_predictions.extend(predictions.tolist())
					
					if return_probabilities:
						probabilities = torch.softmax(logits, dim=1).cpu().numpy()
						all_probabilities.extend(probabilities.tolist())
					
					del input_ids, attention_mask, outputs, logits
					
				except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
					if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
						logger.error("GPU memory issue. Try reducing batch size.")
						clear_gpu_memory()
						raise
					else:
						logger.error(f"Model execution failed: {e}")
						raise
		
		result_data = input_data.copy()
		result_data['prediction'] = all_predictions
		
		if label_mapping:
			predicted_labels = [label_mapping.get(int(pred), f"class_{pred}") for pred in all_predictions]
			result_data['predicted_label'] = predicted_labels
		
		if return_probabilities and all_probabilities:
			num_classes = len(all_probabilities[0])
			for i in range(num_classes):
				result_data[f'prob_class_{i}'] = [prob[i] for prob in all_probabilities]
		
		return result_data

class ModelPredictor:
	def __init__(self, model_path: Path, custom_labels: List[str] = None,
				 use_fp16: bool = None):
		self.model_path = Path(model_path)
		
		if not self._detect_model_type():
			raise ValueError(f"Cannot detect BERT model from {self.model_path}")
		
		try:
			self.predictor = BERTPredictor(self.model_path, use_fp16)
		except Exception as e:
			raise RuntimeError(f"Failed to initialize BERT predictor: {e}") from e
		
		self.label_mapping = self._load_label_mapping(custom_labels)
	
	def _detect_model_type(self) -> bool:
		config_path = self.model_path / 'config.json'
		best_model_path = self.model_path / 'best_model.pt'
		tokenizer_path = self.model_path / 'tokenizer.json'
		
		if config_path.exists() and best_model_path.exists() and tokenizer_path.exists():
			try:
				with open(config_path, 'r') as f:
					config = json.load(f)
				return 'bert_model' in config and 'model_config' in config
			except json.JSONDecodeError:
				pass
		
		return False
	
	def _load_label_mapping(self, custom_labels: List[str] = None) -> Dict[int, str]:
		if custom_labels:
			logger.info(f"Using custom labels: {custom_labels}")
			return {i: label for i, label in enumerate(custom_labels)}
		
		config_path = self.model_path / 'config.json'
		if config_path.exists():
			try:
				with open(config_path, 'r') as f:
					config = json.load(f)
				label_mapping = config.get('label_mapping', {})
				if label_mapping:
					return {int(k): str(v) for k, v in label_mapping.items()}
			except json.JSONDecodeError:
				logger.warning("Invalid JSON in config.json")
		
		logger.warning("No label mapping found, using generic class labels")
		return {}
	
	def predict_chunked(self, input_path: Path, text_col: str, output_dir: Path,
						input_chunksize: Optional[int] = None, prediction_batch_size: int = 128,
						return_probabilities: bool = False, num_workers: int = 4) -> Path:
		
		output_dir.mkdir(parents=True, exist_ok=True)
		chunk_idx = 0
		total_processed = 0
		
		logger.info(f"Processing file in chunks (input_chunksize={input_chunksize})")
		
		total_rows, columns = get_file_info(input_path)
		if total_rows:
			logger.info(f"Input file: {total_rows:,} rows, columns: {columns}")
		else:
			logger.info(f"Input file columns: {columns}")
		
		if text_col not in columns:
			raise ValueError(f"Text column '{text_col}' not found. Available: {columns}")
		
		progress_bar = tqdm(
			desc="Processing chunks",
			total=total_rows,
			unit="samples",
			colour='green',
			leave=False
		) if total_rows else tqdm(desc="Processing chunks", colour='green', leave=False)
		
		for input_chunk in load_dataframe_chunked(input_path, chunksize=input_chunksize):
			if killer.kill_now:
				break
			
			texts = input_chunk[text_col].astype(str).tolist()
			
			try:
				chunk_result = self.predictor.predict_chunk(
					texts,
					input_chunk,
					batch_size=prediction_batch_size,
					return_probabilities=return_probabilities,
					num_workers=num_workers,
					label_mapping=self.label_mapping
				)
				
				chunk_file = output_dir / f"chunk_{chunk_idx:06d}.parquet"
				chunk_result.to_parquet(chunk_file, index=False)
				
				total_processed += len(input_chunk)
				chunk_idx += 1
				
				progress_bar.update(len(input_chunk))
				progress_bar.set_postfix({'processed': f"{total_processed:,}", 'chunks': chunk_idx})
				
			except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
				if "out of memory" in str(e).lower():
					logger.error(f"GPU memory exhausted at chunk {chunk_idx}. Reducing batch size and retrying.")
					prediction_batch_size = max(prediction_batch_size // 2, 4)
					clear_gpu_memory()
					
					chunk_result = self.predictor.predict_chunk(
						texts,
						input_chunk,
						batch_size=prediction_batch_size,
						return_probabilities=return_probabilities,
						num_workers=num_workers,
						label_mapping=self.label_mapping
					)
					
					chunk_file = output_dir / f"chunk_{chunk_idx:06d}.parquet"
					chunk_result.to_parquet(chunk_file, index=False)
					
					total_processed += len(input_chunk)
					chunk_idx += 1
					
					progress_bar.update(len(input_chunk))
					progress_bar.set_postfix({'processed': f"{total_processed:,}", 'chunks': chunk_idx})
				else:
					logger.error(f"Model execution failed at chunk {chunk_idx}: {e}")
					raise
		
		progress_bar.close()
		
		logger.info(f"Chunked processing complete: {total_processed:,} samples in {chunk_idx} chunks")
		return output_dir

def main():
	parser = argparse.ArgumentParser(
		description="Batched BERT inference over a large table of texts",
		formatter_class=ColorHelpFormatter
	)
	
	input_group = parser.add_argument_group('Input/Output')
	input_group.add_argument('-i', '--input', type=Path, required=True,
							help='Input data file path (CSV, Excel, Parquet formats supported)')
	input_group.add_argument('-o', '--output', type=Path, required=True,
							help='Output file path for predictions (format auto-detected from extension)')
	input_group.add_argument('-m', '--model', type=Path, required=True,
							help='Trained model directory (config.json, best_model.pt, tokenizer)')
	input_group.add_argument('--text-col', default='text',
							help='Column name containing text data for prediction (default: text)')
	
	gpu_group = parser.add_argument_group('GPU Configuration')
	gpu_group.add_argument('--fp16', action='store_true',
						   help='Force FP16 inference')
	gpu_group.add_argument('--no-fp16', action='store_true',
						   help='Force disable FP16 mixed precision inference (overrides model configuration)')
	
	memory_group = parser.add_argument_group('Memory Management')
	memory_group.add_argument('--input-chunksize', type=int, default=None,
							 help='Rows per chunk (e.g. 10000 for large files)')
	memory_group.add_argument('--auto-chunk', action='store_true',
							 help='Automatically enable chunking for large files over 100MB with optimal chunk size')
	memory_group.add_argument('--batch-size', type=int, default=128,
							 help='Inference batch size (default: 128)')
	memory_group.add_argument('--temp-dir', type=Path, default=None,
							 help='Custom temporary directory for intermediate chunk files (default: system temp)')
	
	performance_group = parser.add_argument_group('Performance Options')
	performance_group.add_argument('-j', '--jobs', type=int, default=4,
								   help='Number of parallel dataloader worker processes for efficient data loading (default: 4)')
	
	output_group = parser.add_argument_group('Output Configuration')
	output_group.add_argument('--output-format', choices=['csv', 'excel', 'parquet'], default=None,
							 help='Output file format override (default: auto-detect from file extension)')
	output_group.add_argument('--include-probabilities', action='store_true',
							 help='Include prediction probability scores for all classes in output columns')
	output_group.add_argument('--labels', nargs='+', default=None,
							 help='Class names, in label order (e.g. --labels none gain loss)')
	
	debug_group = parser.add_argument_group('Debugging & Logging')
	debug_group.add_argument('-v', '--verbose', action='store_true',
							help='Enable verbose logging with detailed processing information and timing')
	debug_group.add_argument('--debug', action='store_true',
							help='Enable debug logging')
	debug_group.add_argument('--keep-temp', action='store_true',
							help='Keep temporary chunk files after processing for debugging and inspection')
	
	args = parser.parse_args()
	
	if args.debug:
		logger.setLevel(logging.DEBUG)
	elif args.verbose:
		logger.setLevel(logging.INFO)
	
	if args.labels and len(args.labels) == 1 and ',' in args.labels[0]:
		args.labels = [label.strip() for label in args.labels[0].split(',')]
	
	if not args.model.exists():
		logger.error(f"Model directory not found: {args.model}")
		sys.exit(1)
	
	if not args.input.exists():
		logger.error(f"Input file not found: {args.input}")
		sys.exit(1)
	
	use_fp16 = None
	if args.fp16:
		use_fp16 = True
	elif args.no_fp16:
		use_fp16 = False
	
	if args.auto_chunk and args.input_chunksize is None:
		file_size_mb = args.input.stat().st_size / (1024 * 1024)
		if file_size_mb > 100:
			args.input_chunksize = 10000
			logger.info(f"Auto-chunking enabled: file size {file_size_mb:.1f}MB, using chunksize={args.input_chunksize}")
	
	if args.input_chunksize:
		logger.info(f"Chunked processing enabled: {args.input_chunksize:,} rows per chunk")
	
	try:
		logger.info(f"Loading BERT model from {args.model}")
		predictor = ModelPredictor(
			args.model,
			custom_labels=args.labels,
			use_fp16=use_fp16
		)
		
		logger.info("Model loaded successfully")
		logger.info(f"Label mapping: {predictor.label_mapping}")
		
	except Exception as e:
		logger.error(f"Failed to load model: {e}")
		if args.debug:
			import traceback
			traceback.print_exc()
		sys.exit(1)
	
	temp_dir = None
	try:
		if args.temp_dir:
			temp_dir = args.temp_dir
			temp_dir.mkdir(parents=True, exist_ok=True)
		else:
			temp_dir = Path(tempfile.mkdtemp(prefix="bert_predictions_"))
		
		logger.info(f"Using temporary directory: {temp_dir}")
		
		chunk_dir = predictor.predict_chunked(
			args.input,
			args.text_col,
			temp_dir,
			input_chunksize=args.input_chunksize,
			prediction_batch_size=args.batch_size,
			return_probabilities=args.include_probabilities,
			num_workers=args.jobs
		)
		
		if killer.kill_now:
			logger.info("Interrupted during prediction")
			sys.exit(1)
		
		output_format = args.output_format or args.output.suffix.lstrip('.').lower()
		if output_format not in ['csv', 'excel', 'xlsx', 'parquet']:
			output_format = 'csv'
		if output_format == 'xlsx':
			output_format = 'excel'
		
		logger.info(f"Merging chunks and saving to {args.output} (format: {output_format})")
		
		merge_prediction_files(chunk_dir, args.output, output_format)
		
		logger.info("Prediction completed successfully")
		
	except Exception as e:
		logger.error(f"Prediction failed: {e}")
		if args.debug:
			import traceback
			traceback.print_exc()
		sys.exit(1)
	
	finally:
		clear_gpu_memory()
		if temp_dir and temp_dir.exists() and not args.keep_temp:
			try:
				shutil.rmtree(temp_dir)
				logger.info("Temporary files cleaned up")
			except Exception as e:
				logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
	main()