from typing import Optional
from dataclasses import field
from dataclasses import dataclass
from transformers import Seq2SeqTrainingArguments

@dataclass
class Seq2SeqAgreementAttractionTrainingArguments(Seq2SeqTrainingArguments):
	"""
	Arguments added to the Seq2SeqTrainingArguments that are useful for
	evaluating model performance in agreement attraction configurations.
	"""
	predict_identical_until_given_word_number: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Whether to force the model to predict an identical sequence to the target sequence "
			"up to the word number given in the metadata field `predict_identical_until_given_word_number`. "
			"Useful for forcing models to produce the correct preamble to ensure that agreement attraction "
			"can be evaluated rather than NA (if the model produces an incorrect preamble)."
		}
	)
	
	predict_from_given_words_after_identical: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Whether to force the model to predict from a prespecified set of words after predicting "
			"an identical sequence up to that point. The prespecified set of words is given in the metadata "
			"field `predict_from_given_words_after_identical`, which should be a list of strings corresponding "
			"to words that can be predicted after the preamble. Setting this to True will also set `predict_"
			"identical_until_given_word_number` to True."
		}
	)
	
	def __post_init__(self):
		if not self.predict_with_generate:
			self.predict_identical_until_given_word_number = False
			self.predict_from_given_words_after_identical = False
		
		if self.predict_from_given_words_after_identical:
			self.predict_identical_until_given_word_number = True
		
		if self.predict_identical_until_given_word_number:
			self.remove_unused_columns = False
		
		super().__post_init__()