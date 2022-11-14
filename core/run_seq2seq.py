# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/huggingface/transformers/tree/master/examples/seq2seq

"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import os
import re
import sys
import gzip
import glob
import json
import torch
import logging
import transformers

import numpy as np
import pandas as pd
import seaborn as sns

from typing import Optional
from metrics import compute_metrics as run_metrics # workaround since hf already defines a function with this name
from operator import itemgetter
from datasets import load_dataset, load_metric
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass, field
from transformers import (
	AutoConfig,
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	HfArgumentParser,
	MBartTokenizer,
	MBartTokenizerFast,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
	default_data_collator,
	set_seed,
)
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)

MULTILINGUAL_TOKENIZERS = (MBartTokenizer, MBartTokenizerFast)

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""
	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
	)
	
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)
	
	random_weights: bool = field(
		default=False,
		metadata={
			"help": "Randomize weights when loading a model."
		},
	)
	
	random_layers: str = field(
		default=None,
		metadata={"help": "Randomize specific layers of the model. Format: comma-separated list of integers (e.g., `7,8,9` )."
		},
	)

@dataclass
class DataTrainingArguments:
	"""Arguments pertaining to what data we are going to input our model for training and eval."""
	task: str = field(
		default="summarization",
		metadata={
			"help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
			"pegasus) or translation (or translation_{xx}_to_{yy})."
		},
	)
	
	dataset_name: Optional[str] = field(
		default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	
	dataset_config_name: Optional[str] = field(
		default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	
	text_column: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
	)
	
	summary_column: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
	)
	
	train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
	
	do_learning_curve: bool = field(default=False, metadata={"help": "Whether to run predictions on all checkpoints for a learning curve."})
	
	validation_file: Optional[str] = field(
		default=None,
		metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
	)
	
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)
	
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	
	max_source_length: Optional[int] = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	
	max_target_length: Optional[int] = field(
		default=128,
		metadata={
			"help": "The maximum total sequence length for target text after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	
	val_max_target_length: Optional[int] = field(
		default=None,
		metadata={
			"help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
			"This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
			"during ``evaluate`` and ``predict``."
		},
	)
	
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": "Whether to pad all samples to model maximum sentence length. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
			"efficient on GPU but very bad for TPU."
		},
	)
	
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	
	max_val_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
			"value if set."
		},
	)
	
	source_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
	
	target_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
	
	source_prefix: Optional[str] = field(default=None, metadata={"help": "Source prefix for tokenizer (for mBART)"})
	
	target_prefix: Optional[str] = field(default=None, metadata={"help": "Target prefix for tokenizer (for mBART)"})
	
	prefix_from_file: bool = field(
		default=False,
		metadata={
			"help": "Whether to set language ids independently for each example."
		},
	)
	
	eval_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
	
	ignore_pad_token_for_loss: bool = field(
		default=True,
		metadata={
			"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
		},
	)
	
	source_prefix: Optional[str] = field(
		default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
	)
	
	def __post_init__(self):
		if self.dataset_name is None and self.train_file is None and self.validation_file is None:
			raise ValueError("Need either a dataset name or a training/validation file.")
		else:
			if self.train_file is not None:
				extension = self.train_file.split(".")[-1]
			
				if extension == 'gz':
					extension = self.train_file.split('.')[-2]
				
				assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
			
			if self.validation_file is not None:
				extension = self.validation_file.split(".")[-1]
				
				if extension == 'gz':
					extension = self.validation_file.split('.')[-2]
				
				assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
		
		if not self.task.startswith("summarization") and not self.task.startswith("translation"):
			raise ValueError(
				"`task` should be summarization, summarization_{dataset}, translation or translation_{xx}_to_{yy}."
			)
		
		if self.val_max_target_length is None:
			self.val_max_target_length = self.max_target_length

def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.
	
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
	
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	if (
		os.path.exists(training_args.output_dir)
		and os.listdir(training_args.output_dir)
		and training_args.do_train
		and not training_args.overwrite_output_dir
	):
		raise ValueError(
			f"Output directory ({training_args.output_dir}) already exists and is not empty."
			"Use --overwrite_output_dir to overcome."
		)
	
	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
	)
	
	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(training_args.local_rank):
		transformers.utils.logging.set_verbosity_info()
	
	# Set seed before initializing model.
	set_seed(training_args.seed)
	
	# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
	# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
	# (the dataset will be downloaded automatically from the datasets Hub).
	#
	# For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
	# second column for the summaries (unless you specify column names for this with the `text_column` and
	# `summary_column` arguments).
	# For translation, only JSON files are supported, with one field named "translation" containing two keys for the
	# source and target languages (unless you adapt what follows).
	#
	# In distributed training, the load_dataset function guarantee that only one local process can concurrently
	# download the dataset.
	if data_args.dataset_name is not None:
		# Downloading and loading a dataset from the hub.
		datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
	else:
		data_files = {}
		for split, file in zip(['train', 'validation'], [data_args.train_file, data_args.validation_file]):
			if file is not None:
				data_files[split] 	= file
				extension 			= file.split(".")[-1]
				if extension == 'gz':
					extension 		= file.split('.')[-2]
		
		datasets = load_dataset(extension, data_files=data_files)
	
	# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
	# https://huggingface.co/docs/datasets/loading_datasets.html.
	
	# Load pretrained model and tokenizer
	#
	# Distributed training:
	# The .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	model = AutoModelForSeq2SeqLM.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	if model_args.random_weights:
		logger.info("Randomizing weights")
		model.init_weights()
	
	# Set decoder_start_token_id
	if model.config.decoder_start_token_id is None and isinstance(tokenizer, MULTILINGUAL_TOKENIZERS):
		if data_args.target_prefix:
			model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_prefix)
		elif not data_args.prefix_from_file:
			raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
	
	# Get the default prefix if None is passed.
	if data_args.source_prefix is None:
		task_specific_params = model.config.task_specific_params
		if task_specific_params is not None:
			prefix = task_specific_params.get("prefix", "")
		else:
			prefix = ""
	else:
		prefix = data_args.source_prefix
	
	# Preprocessing the datasets.
	# We need to tokenize inputs and targets.
	if training_args.do_train:
		num_rows = datasets['train'].num_rows
		save_steps = max(1,int((num_rows*training_args.num_train_epochs)/(training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps)/15))
		training_args.save_steps = save_steps
		column_names = datasets["train"].column_names
	else:
		column_names = datasets["validation"].column_names
	
	logger.info("Training/evaluation parameters %s", training_args)
	
	# For translation we set the codes of our source and target languages (only useful for mBART, the others will
	# ignore those attributes).
	if data_args.task.startswith("translation"):
		for tokenizer_attr, arg, lang in zip(
											['src_lang', 		'tgt_lang'], 
											['source_prefix', 	'target_prefix'], 
											['source_lang', 	'target_lang']
										):
			if isinstance(tokenizer, MULTILINGUAL_TOKENIZERS):
				if arg in data_args and getattr(data_args, arg):
					setattr(tokenizer, tokenizer_attr, getattr(data_args, arg))
				elif data_args.prefix_from_file:
					setattr(tokenizer, tokenizer_attr, "en_XX") # placeholder
			elif getattr(data_args, lang) is not None:
				setattr(tokenizer, tokenizer_attr, getattr(data_args, lang))
	
	# To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
	# them all).
	source_lang, target_lang, text_column, summary_column = None, None, None, None
	
	# Get the language codes for input/target.
	lang_search = re.match("translation_([a-z]+)_to_([a-z]+)", data_args.task)
	if data_args.source_lang is not None:
		source_lang = data_args.source_lang.split("_")[0]
	else:
		assert (
			lang_search is not None
		), "Provide a source language via --source_lang or rename your task 'translation_xx_to_yy'."
		source_lang = lang_search.groups()[0]
	
	if data_args.target_lang is not None:
		target_lang = data_args.target_lang.split("_")[0]
	else:
		assert (
			lang_search is not None
		), "Provide a target language via --target_lang or rename your task 'translation_xx_to_yy'."
		target_lang = lang_search.groups()[1]
	
	# Temporarily set max_target_length for training.
	max_target_length 	= data_args.max_target_length
	padding 			= "max_length" if data_args.pad_to_max_length else False
	
	def preprocess_function(examples):
		inputs 			= [ex["prefix"] + ex["src"] for ex in examples["translation"]]
		targets 		= [ex["tgt"] for ex in examples["translation"]]

		inputs 			= [prefix + inp for inp in inputs]
		model_inputs 	= tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

		# Setup the tokenizer for targets
		with tokenizer.as_target_tokenizer():
			labels 		= tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

		if data_args.prefix_from_file:
			lang_ids 	= [tokenizer.convert_tokens_to_ids(ex["lang"]) for ex in examples["translation"]]
			for idx, (model_input, label) in enumerate(zip(model_inputs["input_ids"], labels["input_ids"])):
				model_input[-1] = lang_ids[idx]
				label[-1] 		= lang_ids[idx] 
				label 			= [lang_ids[idx]] + label
		
		# If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
		# padding in the loss.
		if padding == "max_length" and data_args.ignore_pad_token_for_loss:
			labels["input_ids"] = [
				[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
			]

		model_inputs["labels"] = labels["input_ids"]
		return model_inputs
	
	if training_args.do_train:
		train_dataset 		= datasets["train"]
		
		if data_args.max_train_samples is not None:
			train_dataset 	= train_dataset.select(range(data_args.max_train_samples))
		
		train_dataset 		= train_dataset.map(
			preprocess_function,
			batched=True,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not data_args.overwrite_cache,
		)
	
	if training_args.do_eval or data_args.do_learning_curve:
		max_target_length 	= data_args.val_max_target_length
		eval_dataset 		= datasets["validation"]
		
		if data_args.max_val_samples is not None:
			eval_dataset 	= eval_dataset.select(range(data_args.max_val_samples))
		
		eval_dataset 		= eval_dataset.map(
			preprocess_function,
			batched=True,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not data_args.overwrite_cache,
		)
	
	# Data collator
	label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
	if data_args.pad_to_max_length:
		data_collator = default_data_collator
	else:
		data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=label_pad_token_id)
	
	# Metric
	metric_name = "rouge" if data_args.task.startswith("summarization") else "sacrebleu"
	metric = load_metric(metric_name)
	
	def compute_metrics(eval_preds):
		preds, labels = eval_preds
		if isinstance(preds, tuple):
			preds = preds[0]
		
		decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
		
		if data_args.ignore_pad_token_for_loss:
			# Replace -100 in the labels as we can't decode them.
			labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
		
		decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

		# Some simple post-processing
		decoded_preds = [pred.strip() for pred in decoded_preds]
		decoded_labels = [label.strip() for label in decoded_labels]
		
		if metric_name == "sacrebleu":
			decoded_labels = [[label] for label in decoded_labels]
		
		result = metric.compute(predictions=decoded_preds, references=decoded_labels)
		
		# Extract a few results from ROUGE
		if metric_name == "rouge":
			result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
		else:
			result = {"bleu": result["score"]}
		
		prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
		result["gen_len"] = np.mean(prediction_lens)
		
		return result
	
	# Initialize our Trainer
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics if training_args.predict_with_generate else None,
	)
	
	# Training
	if training_args.do_train:
		train_result = trainer.train(
			model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
		)
		trainer.save_model() # Saves the tokenizer too for easy upload
		
		output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
		if trainer.is_world_process_zero():
			with open(output_train_file, "w") as writer:
				logger.info("***** Train results *****")
				for key, value in sorted(train_result.metrics.items()):
					logger.info(f"{key} = {value}")
					writer.write(f"{key} = {value}\n")
			
			# Need to save the state, since Trainer.save_model saves only the tokenizer with the model
			trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
	
	# We don't actually use this at all, since accuracy is always zero and predictions are garbo before
	# any fine-tuning.
	# Evaluation
	'''
		results = {}
		if training_args.do_eval:
			logger.info("*** Evaluate ***")
			
			basename = os.path.basename(data_args.validation_file).replace(".json.gz", "")
			output_pred_file = os.path.join(training_args.output_dir, basename + ".eval_preds_seq2seq.txt")
			
			# do not re-compute predictions if they already exist
			if not os.path.exists(output_pred_file):
				predictions = trainer.predict(test_dataset=eval_dataset, max_length=100)
				
				if trainer.is_world_process_zero():
					with open(output_pred_file, "w") as writer:
						for pred in tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True):
							writer.write(pred + "\n")
				
			metrics = run_metrics(output_pred_file, data_args.validation_file)	
			
			if trainer.is_world_process_zero():
				output_eval_file = os.path.join(training_args.output_dir, basename + ".eval_results_seq2seq.txt")
				with open(output_eval_file, "w") as writer:
					for m in metrics:
						display_name = m.replace('_', ' ')
						display_name = display_name[0].upper() + display_name[1:]
						writer.write(f"{display_name} accuracy: {metrics[m]}\n")
	'''
	
	if data_args.do_learning_curve:
		
		basename = os.path.basename(data_args.validation_file).replace(".json.gz", "")
		
		with gzip.open(data_args.validation_file.replace('.json.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
			metadata = [json.loads(l) for l in in_file.readlines()]	
		
		for path in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*", "")):
			output_pred_file = os.path.join(path, basename + ".eval_preds_seq2seq.txt")
			
			# do not re-compute predictions if they already exist
			if not os.path.exists(output_pred_file):
				model = AutoModelForSeq2SeqLM.from_pretrained(
					path,
					from_tf=bool(".ckpt" in model_args.model_name_or_path),
					config=config,
					cache_dir=model_args.cache_dir,
					revision=model_args.model_revision,
					use_auth_token=True if model_args.use_auth_token else None,
				)
				
				trainer = Seq2SeqTrainer(
					model=model,
					args=training_args,
					train_dataset=train_dataset if training_args.do_train else None,
					eval_dataset=eval_dataset if training_args.do_eval else None,
					tokenizer=tokenizer,
					data_collator=data_collator,
					compute_metrics=None,
				)
				
				predictions = trainer.predict(test_dataset=eval_dataset, max_length=100)
				
				if trainer.is_world_process_zero():
					with open(output_pred_file, "w") as writer:
						for pred in tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True):
							writer.write(pred + "\n")
			
			metrics = run_metrics(output_pred_file, data_args, return_results='list')
			
			it_res 	= re.match(".*checkpoint-([0-9]+)[/].*", path)
			it 		= it_res.group(1)
			
			if trainer.is_world_process_zero():
				output_eval_file = os.path.join(path, basename + '.eval_results_seq2seq.csv.gz')
				
				# sort these columns last
				metric_names = list(metrics.keys())
				
				# put all the metrics into a single list of dicts
				metrics = [{k: v for d in ds for k, v in d.items()} for ds in zip(*[metrics[m] for m in metrics], metadata)]
				metrics = pd.DataFrame(metrics)
				metrics.insert(0, 'iteration', it)
				
				max_checkpoint = max([
					int(re.findall('-([0-9]*)$', d)[0]) 
					for d in os.listdir(training_args.output_dir) 
						if d.startswith('checkpoint')
				])
				
				# this is only an approximation, but it's better than the default of 3.0
				num_train_epochs = round((
					max_checkpoint * training_args.per_device_train_batch_size * training_args.per_device_eval_batch_size/
					datasets['train'].num_rows
				) + 0.5)
				
				metrics = metrics.assign(
						model_name=re.sub('["\']', '', model_args.model_name_or_path),
						train_dataset=os.path.basename(data_args.train_file).replace('.json.gz', ''),
						test_dataset=os.path.basename(data_args.validation_file).replace('.json.gz', ''),
						learning_rate=training_args.learning_rate,
						num_train_epochs=num_train_epochs,
						n_training_examples=datasets['train'].num_rows,
						n_test_examples=datasets['validation'].num_rows,
						n_params=f'{round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000)}M'
					)
				
				metrics = metrics[[c for c in metrics.columns if not c in metric_names] + metric_names]
				metrics.to_csv(output_eval_file, index=False, na_rep='NaN')
		
		# plot learning curve
		eval_files 	= [
			os.path.join(path, f'{basename}.eval_results_seq2seq.csv.gz')
			for path in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
		]
		
		eval_preds 	= pd.concat([pd.read_csv(eval_file) for eval_file in eval_files], ignore_index=True)
		eval_preds 	= eval_preds.sort_values('iteration', kind='stable').reset_index(drop=True)
		eval_preds.to_csv(os.path.join(training_args.output_dir, f'{basename}.eval_results_seq2seq.csv.gz'), index=False, na_rep='NaN')
		
		grouping_vars = [
			k 
			for k in list(metadata[0].keys()) 
				if 	not any(x in k for x in ['pos_seq', 'history']) and
					not k in [
						'tense', 
						'each_distractor_structure', 
						'object_number',
					]
		]
		
		# title = os.path.split(training_args.output_dir)
		# title = [s for s in title if s][-1]
		# # model = re.findall('(.*)-finetuning', title)
		# # model = model[0]
		# title = re.findall('finetuning-(.*)-.*?$', title)
		# title = title[0].replace('-', '_', 1)
		title = 'model: ' + re.sub('[\'"]', '', model_args.model_name_or_path)
		title += f'\ntraining: {os.path.basename(data_args.train_file).replace(".json.gz", "")}, '
		title += f'test: {os.path.basename(data_args.validation_file).replace(".json.gz", "")}'
		
		with PdfPages(os.path.join(training_args.output_dir, f'{basename}.learning_curves.pdf')) as pdf:
			common_kwargs = dict(x='iteration', errorbar=None)
			
			# var is None is used for an overall plot without groups
			for var in [None] + grouping_vars:
				# fig, ax = plt.subplots(1, tight_layout=True)
				for c in metric_names:
					# if all values are na, no plot can or should be created
					if not all(eval_preds[c].isna()):
						plot_kwargs = common_kwargs.copy()
						plot_kwargs.update(dict(y=c))
						plot_kwargs.update(dict(
							data = eval_preds[['iteration', c, 'tense']]
								if var is None 
								else eval_preds.assign(
									var = [
										' + '.join([p, str(v)]) 
										if not (isinstance(v,float) and np.isnan(v)) 
										else ' + '.join([p, 'None']) 
											for p, v in zip(eval_preds.tense, eval_preds[var])
									]
								).sort_values(['iteration',var]).reset_index(drop=True)[['iteration', c, 'var']]
						))
						
						# filter to only the stuff that can actually be plotted (bc you can't plot something
						# that doesn't exist!)
						plot_kwargs['data'] = plot_kwargs['data'][(~plot_kwargs['data'][c].isna())].reset_index(drop=True)
						if var is not None:
							plot_kwargs['data'] = plot_kwargs['data'][(~plot_kwargs['data']['var'].isna())].reset_index(drop=True)
						
						if not plot_kwargs['data'].empty:
							plot_kwargs.update(dict(label=c.replace('_', ' ')) if var is None else dict(hue='var'))
							if var is None:
								plot_kwargs.update(dict(style='tense'))
							
							sns.lineplot(**plot_kwargs)
							
							if var is not None or c == metric_names[-1]:
								ax = plt.gca()
								
								legend_kwargs = dict(fontsize=8)
								
								handles, labels = ax.get_legend_handles_labels()
								if var is not None:
									# add count for each condition to legend
									# sometimes a metric is NA because of the model's prediction 
									# at a particular iteration. however, it's impractical to show
									# the counts for each group for each iteration. we choose the 
									# most recent value for the group as a compromise, since this
									# reflects the most recent state of the model's predictions,
									# even though it might be inaccurate for earlier steps
									counts = plot_kwargs['data'].groupby('iteration').value_counts(['var']).groupby(['var']).last()
									counts.index = counts.index.astype(str) # cast to string for boolean and int indices
									labels = [label + f' ($n={counts[label]}$)'for label in labels]
								else:
									# this is for the overall plot
									# since we plot each line one at a time due to the data format, we end up with a lot of
									# duplicated legend entries we don't want. this removes them by filtering to the first instance of each
									# indices = [
									# 			i 
									# 			for i, label in enumerate(labels + [None]) 
									# 				if 	not label == None and 
									# 			   		label == (labels + [None])[i+1] or
									# 			   		(
									# 			   			(label in ['pres', 'past'] and 
									# 			   			i > len(labels)-3)
									# 			   		)
									# 		]
									# get the list of unique labels, and exclude the task prefix variable
									indices = [labels.index(label) for label in list(dict.fromkeys(labels)) if not label in ['pres', 'past']]
									handles = [handles[i] for i in indices]
									labels = [labels[i] for i in indices]
								
								legend_kwargs.update(dict(handles=handles, labels=labels))
								
								ax.legend(**legend_kwargs)
								ax.get_legend().set_title(f'tense + {var.replace("_", " ")}' if var is not None else 'metric')
								
								# set axis ticks and limits for display
								plt.xticks([c for c in plot_kwargs['data'].iteration.unique() if c % 1000 == 0])
								_, ymargin = ax.margins()
								ax.set_ylim((0-ymargin, 1+ymargin))
								
								# set ylabel
								ax.set_ylabel('proportion')
								
								fig = plt.gcf()
								# fig.tight_layout()
								fig.set_size_inches(8, 6)
								suptitle = f'{title}' 
								suptitle += (
									f'\ngroups: tense + {var.replace("_", " ")}\n{c.replace("_", " ")}' 
									if var is not None 
									else ''
								)
								fig.suptitle(suptitle)
								fig.tight_layout()
								# pdf.savefig(bbox_inches='tight')
								pdf.savefig()
								plt.close()
								del fig
						else:
							logger.warning(f'All results of "{c}" are NaN (grouping_vars={var}).')
							logger.warning('No plot will be created.')
							logger.warning('If this is unexpected, check your metric.')
					else:
						logger.warning(f'All results of "{c}" are NaN (grouping_vars={var}).')
						logger.warning('No plot will be created.')
						logger.warning('If this is unexpected, check your metric.')
					
				# del fig
					
if __name__ == '__main__':
	
	main()