import logging
from pytorch_transformers import RobertaTokenizer
from pytorch_transformers.tokenization_roberta import VOCAB_FILES_NAMES, PRETRAINED_VOCAB_FILES_MAP, PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

logger = logging.getLogger(__name__)

class GNLITokenizer(RobertaTokenizer):
	"""
	A Tokenizer derived from the RobertaTokenizer class.

	This extends the RobertaTokenizer by adding in special tokens to represent the labels
	"""
	vocab_files_names = VOCAB_FILES_NAMES
	pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
	max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

	SPECIAL_TOKENS_ATTRIBUTES = [
		"bos_token",
		"eos_token",
		"unk_token",
		"sep_token",
		"pad_token",
		"cls_token",
		"mask_token",
		"entail_token",
		"neutral_token",
		"contradict_token",
		"additional_special_tokens",
	]

	@property
	def entail_token(self):
		if self._entail_token is None:
			logger.error("Using entail_token, but it is not set yet.")
		return self._entail_token

	@property
	def neutral_token(self):
		if self._neutral_token is None:
			logger.error("Using neutral_token, but it is not set yet.")
		return self._neutral_token

	@property
	def contradict_token(self):
		if self._contradict_token is None:
			logger.error("Using contradict_token, but it is not set yet.")
		return self._contradict_token

	@entail_token.setter
	def entail_token(self, value):
		self._entail_token = value

	@neutral_token.setter
	def neutral_token(self, value):
		self._neutral_token = value

	@contradict_token.setter
	def contradict_token(self, value):
		self._contradict_token = value

	@property
	def entail_token_id(self):
		return self.convert_tokens_to_ids(self.entail_token)

	@property
	def neutral_token_id(self):
		return self.convert_tokens_to_ids(self.neutral_token)

	@property
	def contradict_token_id(self):
		return self.convert_tokens_to_ids(self.contradict_token)

	def __init__(
		self,
		vocab_file,
		merges_file,
		errors="replace",
		bos_token="<s>",
		eos_token="</s>",
		sep_token="</s>",
		cls_token="<s>",
		unk_token="<unk>",
		pad_token="<pad>",
		mask_token="<mask>",
		entail_token="<entailment>",
		neutral_token="<neutral>",
		contradict_token="<contradiction>",
		**kwargs
	):
		super(GNLITokenizer, self).__init__(
			vocab_file=vocab_file,
			merges_file=merges_file,
			errors=errors,
			bos_token=bos_token,
			eos_token=eos_token,
			unk_token=unk_token,
			sep_token=sep_token,
			cls_token=cls_token,
			pad_token=pad_token,
			mask_token=mask_token,
			entail_token=entail_token,
			neutral_token=neutral_token,
			contradict_token=contradict_token,
			**kwargs
		)
		self._entail_token = None
		self._neutral_token = None
		self._contradict_token = None
		
		# Add the label tokens to the vocabulary
		label_tokens_dict = {"entail_token": entail_token,
							 "neutral_token": neutral_token,
							 "contradict_token": contradict_token}
		self.add_special_tokens(label_tokens_dict)

		self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
		self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens