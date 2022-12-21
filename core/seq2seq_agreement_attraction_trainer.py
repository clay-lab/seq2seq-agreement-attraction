# seq2seq_agreement_attraction_trainer.py
# Created by Michael Wilson
import copy
import torch

from torch import nn
from typing import *
from functools import cached_property
from transformers import Seq2SeqTrainer

class Seq2SeqAgreementAttractionTrainer(Seq2SeqTrainer):
	"""
	Extend the Seq2SeqTrainer to allow for use of two additional
	arguments: `predict_identical_until_given_word_number` and 
	`predict_from_given_words_after_identical`. The first constrains
	predictions to be identical up to and including the word number
	given in the field of the same name in the dataset.
	The second constraints predictions to continue from this point with
	one of the words in the given list. After this, predictions
	may end unconstrained.
	
	This is to aid in evaluating model performance on tasks that probe
	agreement attraction. If the model doesn't produce an identical
	preamble, it can be hard to evaluate its performance. By constraining
	the preamble, we eliminate this variability. In addition, if the model
	doesn't produce a verb (or one of the expected verbs) after the preamble,
	we cannot evaluate whether it predicts agreement attraction or not. By
	restricting predictions to a set of verbs given in the
	`predict_from_given_words_after_identical` field in the dataset,
	we can force the model to predict a verb that will tell us whether
	it shows agreement attraction for each example.
	"""
	@cached_property
	def start_word_prefix(self) -> str:
		'''The start word prefix of the tokenizer.'''
		return chr(9601)
	
	@cached_property
	def start_word_ids(self) -> List[int]:
		'''
		The start word ids in the tokenizer's vocabulary.
		'''
		return [v for k, v in self.tokenizer.get_vocab().items() if k.startswith(self.start_word_prefix)]
	
	@cached_property
	def all_token_ids(self) -> List[int]:
		'''
		A list of all token ids in the model's vocabulary.
		'''
		return list(self.tokenizer.get_vocab().values())
	
	def prefix_allowed_tokens_fn_factory(
		self,
		inputs: Dict[str, Union[torch.Tensor, Any]],
	) -> Callable[[int, torch.Tensor], List[int]]:
		'''
		Returns a function that constrains the output generation to be identical
		up until `predict_identical_until_given_word_number` for each input. If
		`self.args.predict_from_given_Words_after_identical` is set, further
		restricts predictions to be from that list of words. After that,
		any token can be predicted.
		'''
		if not self.args.predict_identical_until_given_word_number:
			return lambda batch_id, input_ids: self.all_token_ids
		
		# we can't decode the labels when the pad token has been replaced with
		# -100 (which HF does so it gets ignored in the loss calculation)
		inputs = copy.deepcopy(inputs)
		inputs['labels'] = [
			[(l if l != -100 else self.tokenizer.pad_token_id) for l in label] for label in inputs['labels']
		]
		
		sentences = self.tokenizer.batch_decode(inputs['labels'])
		identical_until_word_numbers = inputs['predict_identical_until_given_word_number']
		truncated_sentences = []
		for sentence, word_number in zip(sentences, identical_until_word_numbers):
			truncated_sentences.append(' '.join(sentence.split()[:word_number]))
		
		truncated_sequences = self.tokenizer(truncated_sentences)['input_ids']
		truncated_sequences = [[
			[token_id] for token_id in sequence[:sequence.index(self.tokenizer.eos_token_id)]] 
			for sequence in truncated_sequences
		]
		
		if self.args.predict_from_given_words_after_identical:
			next_word_options = inputs['predict_from_given_words_after_identical']
			next_token_options = [
				self.tokenizer(next_word_option, padding=True)['input_ids']
				for next_word_option in next_word_options
			]
			
			# strip off the last part token, which is padding we don't want to be identical
			next_token_options = [
				[next_token[:-1] for next_token in next_token_option] 
				for next_token_option in next_token_options
			]
			
			# zip up the possibilities to add to the sequences
			next_token_options = [
				[list(set(x)) for x in zip(*next_token_option)]
				for next_token_option in next_token_options
			]
			
			# if the allowed words have unequal numbers of tokens,
			# we need to pad in the following way:
			# 	(i) if the position is a position immediately after the
			#		end of the preceding possible word (the next token is the eos_token_id), 
			#		we should predict a new word, or the continuation of the previous word
			#  (ii) if the position is more than 0 away from the previous word,
			#		(the next token is the pad_token_id), we have already either begun predicting
			#		a new word, or continued the previous one. So we must be able to predict anything
			# 
			# This might fail if a word is 3 tokens long, because we could predict the first two parts
			# and the other is 1 token long, because we could allow for the model to predict the 
			# first two subparts of the 3-token word, and then a different final token. It seems
			# likely this will be enough of an edge case, and is difficult to deal with without
			# artificially restricting the rest of the continuation, that we'll just ignore
			# it for now. It would be something we'd need to fix in the prefix_allowed_tokens_fn,
			# since we'd need to check the previously predicted id, and decide what to do based on that.
			next_token_options = [
				[
					list(set(
						[option for option in options 
							if not option in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
						] +
						(
							self.start_word_ids if any(option == self.tokenizer.eos_token_id for option in options) else []
						) +
						(
							self.all_token_ids if any(option == self.tokenizer.pad_token_id for option in options) else []
						)
					))
					for options in next_token_option
				]
				for next_token_option in next_token_options
			]
			
			for truncated_sequence, next_token_option in zip(truncated_sequences, next_token_options):
				truncated_sequence += next_token_option
		
		def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
			'''
			Determines which tokens can be predicted at each decoding step, according to the 
			metadata and general constraints (i.e., mask span token must be first).
			'''
			if (len(input_ids) - 1) <= (len(truncated_sequences[batch_id]) - 1):
				return truncated_sequences[batch_id][len(input_ids)-1]
			
			return self.all_token_ids
		
		return prefix_allowed_tokens_fn
	
	def prediction_step(
		self,
		model: nn.Module,
		inputs: Dict[str, Union[torch.Tensor, Any]],
		prediction_loss_only: bool,
		ignore_keys: Optional[List[str]] = None,
	) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
		"""
		Perform an evaluation step on `model` using `inputs`.
		Injects a function that allows predictions to start with only
		the values in `inputs['predict_identical_until_given_word_number']`
		and continue with the values in `inputs['predict_from_given_words_after_identical']`.
		
		Args:
			model (`nn.Module`):
				The model to evaluate.
			inputs (`Dict[str, Union[torch.Tensor, Any]]`):
				The inputs and targets of the model.
				The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
				argument `labels`. Check your model's documentation for all accepted arguments.
			prediction_loss_only (`bool`):
				Whether or not to return the loss only.
		Return:
			Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
			labels (each being optional).
		"""
		if not self.args.predict_identical_until_given_word_number:
			if 'predict_identical_until_given_word_number' in inputs:
				del inputs['predict_identical_until_given_word_number']
			
			if 'predict_from_given_words_after_identical' in inputs:
				del inputs['predict_from_given_words_after_identical']
			
			return super().prediction_step(
				model=model, inputs=inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
			)
		
		prefix_allowed_tokens_fn = self.prefix_allowed_tokens_fn_factory(inputs=inputs)
		del inputs['predict_identical_until_given_word_number']
		del inputs['predict_from_given_words_after_identical']
		
		original_gen_kwargs = self._gen_kwargs.copy()
		self._gen_kwargs.update({'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn})
		
		predictions = super().prediction_step(
			model=model, inputs=inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
		)
		
		self._gen_kwargs = original_gen_kwargs
		
		return predictions
