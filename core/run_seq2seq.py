# This is a heavily refactored version of run_seq2seq.py by Huggingface,
# modified by Michael Wilson (12/15/2022). The original comment and copyright
# from Huggingface are below.
# ----------------------------------------------------------------------------------------------------------------
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
import gzip
import glob
import json
import torch
import logging
import transformers

import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from datasets import load_dataset, Dataset, DatasetDict
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from transformers import (
	AutoConfig,
	AutoModelForSeq2SeqLM,
	AutoTokenizer,
	HfArgumentParser,
	set_seed,
)
from transformers.trainer_utils import is_main_process

from metrics import compute_metrics
from constants import *
from model_arguments import ModelArguments
from data_training_arguments import DataTrainingArguments
from seq2seq_agreement_attraction_trainer import Seq2SeqAgreementAttractionTrainer
from data_collator_for_seq2seq_agreement_attraction import DataCollatorForSeq2SeqAgreementAttraction
from seq2seq_agreement_attraction_training_arguments import Seq2SeqAgreementAttractionTrainingArguments

from safe_shared_temporary_directory import SafeSharedTemporaryDirectory

logger = logging.getLogger(__name__)

def sort_human(l: List[str]) -> List[str]:
	'''
	Sort file names with numbers in a human-like way, instead of as strings.
	From https://stackoverflow.com/questions/3426108/how-to-sort-a-list-of-strings-numerically
	
		params:
			l (List[str]): a list of file names that contain non-zero padded numbers to sort
		
		returns:
			the list of file names, sorted in a human-like way
	'''
	alphanum = lambda key: int(re.findall('([0-9]*)$', key[:-1])[0])
	return sorted(l, key=alphanum)

def parse_arguments() -> Tuple:
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.
	parser = HfArgumentParser((
		ModelArguments, 
		DataTrainingArguments, 
		Seq2SeqAgreementAttractionTrainingArguments
	))
	
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	
	if data_args.do_learning_curve:
		training_args.do_eval = True
	
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
	
	return model_args, data_args, training_args

def setup_logging(training_args: Seq2SeqAgreementAttractionTrainingArguments) -> None:
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
	
	logger.info(f'Training/evaluation parameters:\n{training_args}')
	
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(training_args.local_rank):
		transformers.utils.logging.set_verbosity_info()

def load_datasets(data_args: DataTrainingArguments) -> DatasetDict:
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
	# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
	# https://huggingface.co/docs/datasets/loading_datasets.html.
	if data_args.dataset_name is not None:
		# Downloading and loading a dataset from the hub.
		return load_dataset(data_args.dataset_name, data_args.dataset_config_name)
	
	data_files = {}
	if data_args.train_file is not None:
		extension 			= data_args.train_file.split('.')[-1]
		if extension == 'gz':
			extension 		= data_args.train_file.split('.')[-2]
		
		return load_dataset(extension, data_files={'train': data_args.train_file})
	
	if data_args.validation_file is not None:
		extension 			= data_args.validation_file.split('.')[-1]
		if extension == 'gz':
			extension 		= data_args.validation_file.split('.')[-2]
		
		with gzip.open(data_args.validation_file.replace('.json', '_metadata.json'), 'rt') as in_file:
			metadata = [json.loads(l.strip()) for l in in_file.readlines()]
		
		metadata = [{
			k: v for k, v in m.items() 
			if k in [
				'predict_identical_until_given_word_number', 
				'predict_from_given_words_after_identical'
			]
		} for m in metadata]
		
		if not any(metadata):
			return load_dataset(extension, data_files={'validation': data_args.validation_file})
		
		with gzip.open(data_args.validation_file, 'rt') as in_file:
			dataset = [json.loads(l.strip()) for l in in_file.readlines()]
		
		for d, m in zip(dataset, metadata):
			d['translation'].update(**m)
		
		# we load this way to avoid writing out a temp file,
		# which causes issues since it'll try to load from the cache
		return DatasetDict({'validation': Dataset.from_pandas(pd.DataFrame(data=dataset))})

def load_config_tokenizer_model(
	model_args: ModelArguments, 
	training_args: Seq2SeqAgreementAttractionTrainingArguments,
	config: AutoConfig = None,
	tokenizer: AutoTokenizer = None,
	model: AutoModelForSeq2SeqLM = None,
	model_path: str = None
) -> Tuple:
	# Load pretrained model and tokenizer
	#
	# Distributed training:
	# The .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	
	if config is None:
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	
	if tokenizer is None:
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	
	if model is None:
		if model_path is not None:
			model_args.model_name_or_path = model_path
		
		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_args.model_name_or_path,
			from_flax=bool(
						os.path.isdir(model_args.model_name_or_path) and 
						'flax_model.msgpack' in os.listdir(model_args.model_name_or_path)
					),
			from_tf=bool('.ckpt' in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		
	# Set seed before initializing model.
	set_seed(training_args.seed)
	if model_args.random_weights:
		logger.info('Randomizing weights')
		model.init_weights()
	
	return config, tokenizer, model

def get_task_prefix(
	data_args: DataTrainingArguments, 
	model: AutoModelForSeq2SeqLM
) -> str:
	# Get the default prefix if None is passed.
	if data_args.source_prefix is None:
		task_specific_params = model.config.task_specific_params
		if task_specific_params is not None:
			prefix = task_specific_params.get('prefix', '')
		else:
			prefix = ''
	else:
		prefix = data_args.source_prefix
	
	return prefix

def prepare_datasets(
	datasets: Dataset, 
	prefix: str, 
	training_args: Seq2SeqAgreementAttractionTrainingArguments, 
	data_args: DataTrainingArguments,
	tokenizer: AutoTokenizer,
) -> Tuple:
	# Preprocessing the datasets.
	# We need to tokenize inputs and targets.
	if training_args.do_train:
		column_names = datasets['train'].column_names
	else:
		column_names = [
			column_name for column_name in datasets['validation'].column_names
			if not column_name in [
				'predict_identical_until_given_word_number', 
				'predict_from_given_words_after_identical'
			]
		]
	
	# Temporarily set max_target_length for training.
	max_target_length 	= data_args.max_target_length
	padding 			= 'max_length' if data_args.pad_to_max_length else False
	
	def preprocess_function(examples: Dict[str, Dict[str,str]]) -> Dict[str, torch.Tensor]:
		inputs 			= [ex['prefix'] + ex['src'] for ex in examples['translation']]
		targets 		= [ex['tgt'] for ex in examples['translation']]
		
		inputs 			= [prefix + inp for inp in inputs]
		model_inputs 	= tokenizer(
							inputs,
							max_length=data_args.max_source_length, 
							padding=padding, 
							truncation=True
						)
		
		with tokenizer.as_target_tokenizer():
			model_inputs['labels'] = tokenizer(
										targets, 
										max_length=max_target_length, 
										padding=padding,
										truncation=True
									)['input_ids']
		
		# If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
		# padding in the loss.
		if padding == 'max_length' and data_args.ignore_pad_token_for_loss:
			model_inputs['labels'] = [
				[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs['labels']
			]
		
		if not training_args.do_train:
			if any('predict_identical_until_given_word_number' in ex for ex in examples['translation']):
				model_inputs['predict_identical_until_given_word_number'] = [
					ex.get('predict_identical_until_given_word_number', 0) for ex in examples['translation']
				]
			
			if any('predict_from_given_words_after_identical' in ex for ex in examples['translation']):
				model_inputs['predict_from_given_words_after_identical'] = [
					ex.get('predict_from_given_words_after_identical', None) for ex in examples['translation']
				]
		
		return model_inputs
	
	if training_args.do_train:
		dataset 	= datasets['train']
		
		if data_args.max_train_samples is not None:
			dataset = dataset.select(range(data_args.max_train_samples))
	
	if training_args.do_eval or data_args.do_learning_curve:
		max_target_length 	= data_args.val_max_target_length
		dataset 			= datasets['validation']
		
		if data_args.max_val_samples is not None:
			dataset = dataset.select(range(data_args.max_val_samples))
	
	dataset = dataset.map(
		preprocess_function,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
		remove_columns=column_names,
		load_from_cache_file=not data_args.overwrite_cache,
	)
	
	if training_args.do_train:
		setattr(dataset, 'name', os.path.basename(data_args.train_file).replace('.json.gz', ''))
	elif training_args.do_eval or training_args.do_learning_curve:
		setattr(dataset, 'name', os.path.basename(data_args.validation_file).replace('.json.gz', ''))
	
	label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
	data_collator = DataCollatorForSeq2SeqAgreementAttraction(tokenizer, label_pad_token_id=label_pad_token_id)
	
	return dataset, data_collator

def setup_trainer(
	model: AutoModelForSeq2SeqLM,
	tokenizer: AutoTokenizer,
	training_args: Seq2SeqAgreementAttractionTrainingArguments,
	dataset: Dataset,
	data_collator: DataCollatorForSeq2SeqAgreementAttraction,
) -> Seq2SeqAgreementAttractionTrainer:
	'''Sets up and return the Seq2SeqAgreementAttractionTrainer.'''
	if training_args.do_train:
		total_batch_size 	= training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
		updates_per_epoch 	= int(dataset.num_rows/total_batch_size)
		if training_args.max_steps == -1:
			total_steps 		= updates_per_epoch * training_args.num_train_epochs
			save_steps 			= max(1, int(total_steps/15))
		else:
			training_args.num_train_epochs = training_args.max_steps/updates_per_epoch
			save_steps = max(1, int(training_args.max_steps/15))
		
		training_args.save_steps = save_steps
	
	# Initialize our Trainer
	trainer = Seq2SeqAgreementAttractionTrainer(
				model=model,
				args=training_args,
				train_dataset=dataset if training_args.do_train else None,
				eval_dataset=dataset if training_args.do_eval else None,
				tokenizer=tokenizer,
				data_collator=data_collator,
			)
	
	return trainer

def run_train(trainer: Seq2SeqAgreementAttractionTrainer) -> None:
	'''
	Runs training and saves some information used during eval to disk.
	'''
	try:
		# allows us to continue training a stored checkpoint
		train_result = trainer.train(
			model_path=trainer.model.name_or_path if os.path.isdir(trainer.model.name_or_path) else None
		)
	except ValueError:
		# allows us to train a local model which does not have a checkpoint saved
		train_result = trainer.train()
	
	trainer.save_model()
	
	output_train_file = os.path.join(trainer.args.output_dir, 'train_results.json')
	if trainer.is_world_process_zero():
		# Need to save the state, since Trainer.save_model saves only the tokenizer with the model
		trainer.state.save_to_json(os.path.join(trainer.args.output_dir, 'trainer_state.json'))
		metrics = dict(
						sorted({
							**train_result.metrics, 
							'num_train_epochs': trainer.args.num_train_epochs,
							'num_train_examples': trainer.train_dataset.num_rows,
							'train_dataset': trainer.train_dataset.name,
							'learning_rate': trainer.args.learning_rate,
							'n_params': f'{round(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)/1000000)}M'
						}.items())
					)
		
		with open(output_train_file, 'w') as writer:
			json.dump(metrics, writer, indent=4)
			logger.info('***** Train results *****')
			logger.info(f'{metrics}')

def run_eval(
	model_args: ModelArguments,
	data_args: DataTrainingArguments, 
	training_args: Seq2SeqAgreementAttractionTrainingArguments, 
	dataset: Dataset,
	data_collator: DataCollatorForSeq2SeqAgreementAttraction,
	config: AutoConfig,
	tokenizer: AutoTokenizer,
) -> None:
	'''
	Runs evaluation on model predictions for each checkpoint.
	If set, call plot_learning_curves to graph performance
	at each weight update for each metric.
	'''
	basename = os.path.basename(data_args.validation_file).replace('.json.gz', '')
	
	with gzip.open(data_args.validation_file.replace('.json.gz', '_metadata.json.gz'), 'rt', encoding='utf-8') as in_file:
		metadata = [json.loads(l) for l in in_file.readlines()]
	
	for path in sort_human(glob.glob(os.path.join(training_args.output_dir, 'checkpoint-*', ''))):
		output_pred_file = os.path.join(path, basename + '.eval_preds_seq2seq.txt')
		
		# do not re-compute predictions if they already exist
		if not os.path.exists(output_pred_file):
			_, _, model = load_config_tokenizer_model(
						model_args=model_args, 
						training_args=training_args, 
						config=config, 
						tokenizer=tokenizer,
						model_path=path
					)
			
			trainer = setup_trainer(
						model=model,
						tokenizer=tokenizer,
						training_args=training_args,
						dataset=dataset,
						data_collator=data_collator
					)
			no_trainer = False
			
			predictions = trainer.predict(test_dataset=dataset, max_length=data_args.val_max_target_length)
			
			if trainer.is_world_process_zero():
				with open(output_pred_file, 'w') as writer:
					for pred in tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True):
						writer.write(f'{pred}\n')
		else:
			no_trainer = True
		
		metrics = compute_metrics(
			pred_file=output_pred_file, 
			gold_file=data_args.validation_file, 
			return_results='list',
			predict_identical_until_given_word_number=training_args.predict_identical_until_given_word_number,
		)
		
		iteration_match = re.findall('.*checkpoint-([0-9]+)$', path[:-1])[0]
		iteration 		= int(iteration_match)
		
		output_eval_file = os.path.join(path, basename + '.eval_results_seq2seq.csv.gz')
		
		if no_trainer or trainer.is_world_process_zero():
			# sort these columns last
			metric_names = list(metrics.keys())
			
			# put all the metrics into a single list of dicts
			metrics = [{k: v for d in ds for k, v in d.items()} for ds in zip(*[metrics[m] for m in metrics], metadata)]
			metrics = pd.DataFrame(metrics)
			metrics.insert(0, 'iteration', iteration)
			
			output_train_file = os.path.join(training_args.output_dir, 'train_results.json')
			with open(output_train_file, 'r') as in_file:
				train_params = json.load(in_file)
			
			train_dataset = train_params['train_dataset']
			num_train_epochs = train_params['num_train_epochs']
			num_train_examples = train_params['num_train_examples']
			learning_rate = train_params['learning_rate']
			n_params = train_params['n_params']
			
			metrics = metrics.assign(
					model_name=re.sub('["\']', '', model_args.model_name_or_path),
					train_dataset=train_dataset,
					test_dataset=os.path.basename(data_args.validation_file).replace('.json.gz', ''),
					learning_rate=learning_rate,
					num_train_epochs=num_train_epochs,
					n_training_examples=num_train_examples,
					n_test_examples=dataset.num_rows,
					n_params=n_params,
				)
			
			metrics = metrics[[c for c in metrics.columns if not c in metric_names] + metric_names]
			metrics.to_csv(output_eval_file, index=False, na_rep='NaN')
	
	# combine results files into one
	eval_files 	= [
		os.path.join(path, f'{basename}.eval_results_seq2seq.csv.gz')
		for path in glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
	]
	
	eval_preds 	= pd.concat([pd.read_csv(eval_file) for eval_file in eval_files], ignore_index=True)
	eval_preds 	= eval_preds.sort_values('iteration', kind='stable').reset_index(drop=True)
	
	# it might seem redundant to not load this file if it exists
	# but this makes it more convenient to rerun if we add or modify
	# a metric rather than having to add an overwrite argument
	# or manually delete the files each time
	eval_preds.to_csv(os.path.join(training_args.output_dir, f'{basename}.eval_results_seq2seq.csv.gz'), index=False, na_rep='NaN')
	
	if data_args.do_learning_curve:
		grouping_vars = [
				k 
				for k in list(metadata[0].keys()) 
					if 	not any(x in k for x in ['pos_seq', 'history', 'predict']) and
						not k in [
							'tense', 
							'each_distractor_structure', 
							'object_number',
						]
			]
		
		plot_learning_curves(
			directory=training_args.output_dir, 
			metric_names=metric_names,
			eval_preds=eval_preds, 
			grouping_vars=grouping_vars
		)

def plot_learning_curves(
	directory: str, 
	metric_names: List[str],
	eval_preds: pd.DataFrame, 
	grouping_vars: List = None
) -> None:
	'''
	Plots learning curves for eval_preds for each metric in metric_names.
	Separate plots are created for the overall learning curves, as well
	as indivdual plots for each grouping variable in grouping_vars.
	A PDF containing the plots is saved in directory.
	'''
	def format_save_plot(
		title: str, 
		var: str, 
		c: str, 
		xticks: List[int],
		data: pd.DataFrame, 
		pdf: PdfPages
	) -> None:
		'''
		Formats and saves the current plot.
		This handles removing redundant legend information and string formatting,
		setting axis labels and limits, setting plot size, and saving the plot.
		'''
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
			counts = data.groupby('iteration').value_counts(['var']).groupby(['var']).last()
			counts.index = counts.index.astype(str) # cast to string for boolean and int indices
			labels = [label + f' ($n={counts[label]}$)'for label in labels]
		else:
			# this is for the overall plot
			# since we plot each line one at a time due to the data format, we end up with a lot of
			# duplicated legend entries we don't want.
			indices = [labels.index(label) for label in list(dict.fromkeys(labels)) if not label in ['pres', 'past']]
			handles = [handles[i] for i in indices]
			labels = [labels[i] for i in indices]
		
		legend_kwargs.update(dict(handles=handles, labels=labels))
		ax.legend(**legend_kwargs)
		ax.get_legend().set_title(f'tense + {var.replace("_", " ")}' if var is not None else 'metric')
		
		# set axis ticks and limits for display
		plt.xticks(xticks)
		_, ymargin = ax.margins()
		ax.set_ylim((0-ymargin, 1+ymargin))
		
		# set ylabel
		ax.set_ylabel('proportion')
		
		fig = plt.gcf()
		fig.set_size_inches(8, 6)
		suptitle = f'{title}' 
		suptitle += (
			f'\ngroups: tense + {var.replace("_", " ")}\n{c.replace("_", " ")}' 
			if var is not None 
			else ''
		)
		fig.suptitle(suptitle)
		fig.tight_layout()
		pdf.savefig()
		plt.close()
		del fig
	
	# var is None is used for an overall plot without groups
	grouping_vars = [None] + grouping_vars if grouping_vars is not None else [None]
	
	title = 'model: ' + re.sub('[\'"]', '', eval_preds.model_name.unique()[0])
	title += f'\ntraining: {eval_preds.train_dataset.unique()[0]}'
	title += f'\ntest: {eval_preds.test_dataset.unique()[0]}'
	
	with PdfPages(os.path.join(directory, f'{eval_preds.test_dataset.unique()[0]}.learning_curves.pdf')) as pdf:
		common_kwargs = dict(x='iteration', errorbar=None)
		
		for var in grouping_vars:
			something_on_plot = False
			for c in metric_names:
				something_to_plot = True				
				plot_kwargs = common_kwargs.copy()
				plot_kwargs.update(dict(
					y=c,
					data=eval_preds[['iteration', c, 'tense']]
						if var is None 
						else eval_preds.assign(
							var=[
								' + '.join([p, str(v) if not str(v) == 'nan' else 'None']) 
								for p, v in zip(eval_preds.tense, eval_preds[var])
							]
						).sort_values(['iteration', var]).reset_index(drop=True)[['iteration', c, 'var']]
				))
				
				# filter out na values
				plot_kwargs['data'] = plot_kwargs['data'][(~plot_kwargs['data'][c].isna())].reset_index(drop=True)
				if var is not None:
					plot_kwargs['data'] = plot_kwargs['data'][(~plot_kwargs['data']['var'].isna())].reset_index(drop=True)
				
				if plot_kwargs['data'].empty:
					logger.warning(f'All results of "{c}" are NaN (grouping_vars={var}).')
					logger.warning('No plot will be created.')
					logger.warning('If this is unexpected, check your metric.')
					something_to_plot = False
				
				if something_to_plot:
					plot_kwargs.update(dict(label=c.replace('_', ' ')) if var is None else dict(hue='var'))
					if var is None:
						plot_kwargs.update(dict(style='tense'))
					
					sns.lineplot(**plot_kwargs)
					something_on_plot = True
				
				# we want to save the plot in the following cases:
				# 	(i) the grouping variable is not None (i.e., we're not on
				#		the overall plot), and there was something to plot
				#  (ii) the grouping variable is None (we're on the overall plot)
				#		we're on the last step of that plot, and something was added
				#		to the plot in a previous loop
				# otherwise, the plot is blank and we don't want to save a blank page
				if (
					(var is not None and something_to_plot) or 
					(var is None and c == metric_names[-1] and something_on_plot)
				):
					format_save_plot(
						title=title, 
						var=var, 
						c=c, 
						data=plot_kwargs['data'], 
						xticks=eval_preds.iteration.unique(), 
						pdf=pdf
					)

def run_seq2seq() -> None:
	'''
	Main function. Handles loading model, tokenizer, datasets,
	and running training or evaluation depending on passed argument.
	'''
	model_args, data_args, training_args = parse_arguments()
	setup_logging(training_args=training_args)
	config, tokenizer, model = load_config_tokenizer_model(model_args=model_args, training_args=training_args)
	
	datasets = load_datasets(data_args=data_args)
	dataset, data_collator = prepare_datasets(
								datasets=datasets,
								prefix=get_task_prefix(data_args=data_args, model=model),
								training_args=training_args,
								data_args=data_args,
								tokenizer=tokenizer
							)
	
	if training_args.do_train:
		run_train(
			trainer=setup_trainer(
				model=model,
				tokenizer=tokenizer,
				training_args=training_args,
				dataset=dataset,
				data_collator=data_collator
			)
		)
	
	if training_args.do_eval:
		run_eval(
			model_args=model_args,
			data_args=data_args,
			training_args=training_args,
			dataset=dataset,
			data_collator=data_collator,
			config=config,
			tokenizer=tokenizer,
		)

if __name__ == '__main__':
	run_seq2seq()