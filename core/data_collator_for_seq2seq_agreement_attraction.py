import torch

from typing import List, Dict, Any, Union
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq

@dataclass
class DataCollatorForSeq2SeqAgreementAttraction(DataCollatorForSeq2Seq):
	'''
	Data collator that will dynamically pad the inputs received, as well as the labels.
	Does not require everything returned to be a tensor.
	'''
	def __call__(self, features: List[Dict[str, Union[torch.Tensor, Any]]], return_tensors: str = None):
		# we want to allow everything that can be converted to a tensor (i.e.,
		# list of ints) to behave as before
		tokenizer_keys = set(
			k for feature in features for k in feature
			if isinstance(feature[k], list) and all(isinstance(v, (int, float)) for v in feature[k])
		)
		
		non_tokenizer_keys = set(
			k for feature in features for k in feature
			if not (isinstance(feature[k], list) and all(isinstance(v, (int, float)) for v in feature[k]))
		)
		
		# extract out the things that cannot be tokenized/converted to tensors,
		# convert to a dict of lists for readding later
		non_tokenizer_features = {
			k: [feature.get(k) for feature in features] 
			for k in non_tokenizer_keys
		}
		
		tokenizer_features = [
			{k: v for k, v in feature.items() if k in tokenizer_keys} 
			for feature in features
		]
		
		tokenizer_features = super().__call__(features=tokenizer_features, return_tensors=return_tensors)
		
		features = {**tokenizer_features, **non_tokenizer_features}
		
		return features