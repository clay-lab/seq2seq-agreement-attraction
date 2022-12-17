from dataclasses import field
from dataclasses import dataclass

@dataclass
class DataTrainingArguments:
	"""Arguments pertaining to what data we are going to input our model for training and eval."""
	dataset_name: Optional[str] = field(
		default=None, 
		metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	
	dataset_config_name: Optional[str] = field(
		default=None,
		metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	
	train_file: Optional[str] = field(
		default=None, 
		metadata={"help": "The input training data file (a json.gz file)."}
	)
	
	do_learning_curve: Optional[bool] = field(
		default=False, 
		metadata={"help": "Whether to plot a learning curve."}
	)
	
	validation_file: Optional[str] = field(
		default=None,
		metadata={"help": "An evaluation data file to evaluate model performance on (a json.gz file)."},
	)
	
	overwrite_cache: bool = field(
		default=False, 
		metadata={"help": "Overwrite the cached training and evaluation sets"}
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
			"help": "The maximum total sequence length for validation target text after tokenization. "
			"Sequences longer than this will be truncated, sequences shorter will be padded. Will "
			"default to `max_target_length`. This argument is also used to override the ``max_length`` "
			"param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
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
		
		if self.train_file is not None:
			extension = self.train_file.split(".")[-1]
		
			if extension == 'gz':
				extension = self.train_file.split('.')[-2]
			
			if not extension in ['csv', 'json']:
				raise ValueError("`train_file` should be a csv or a json file.")
		
		if self.validation_file is not None:
			extension = self.validation_file.split('.')[-1]
			
			if extension == 'gz':
				extension = self.validation_file.split('.')[-2]
			
			if not extension in ['csv', 'json']:
				raise ValueError("`validation_file` should be a csv or a json file.")
	
		if self.val_max_target_length is None:
			self.val_max_target_length = self.max_target_length