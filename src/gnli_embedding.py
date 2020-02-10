# from copy import copy
# import logging
# import torch
# from torch.nn import Embedding, Module

# logger = logging.getLogger(__name__)

# class GNLIEmbedding(Module):
# 	""" 
# 	Wraps the normal token embeddings and creates embeddings for the labels.
# 	During the `forward()` call, embeddings for tokens are added to embeddings for the label.

# 	This is useful for not having to modify the underlying pretrained BART model.
# 	"""
# 	@property
# 	def num_token_embeddings(self):
# 		return self.embed_tokens.num_embeddings

# 	@property
# 	def num_label_embeddings(self):
# 		return self.embed_labels.num_embeddings

# 	@property
# 	def padding_idx(self):
# 		return self.embed_tokens.padding_idx

# 	@property
# 	def embedding_dim(self):
# 		return self.embed_tokens.embedding_dim

# 	@property
# 	def weight(self):
# 		""" Needed for calculating logits """
# 		return self.embed_tokens.weight

# 	def __init__(self, embed_tokens: Embedding, num_labels: int):
# 		super(GNLIEmbedding, self).__init__()
# 		self.embed_tokens = embed_tokens
# 		self.embed_labels = Embedding(num_labels, embed_tokens.embedding_dim)
		
# 	def forward(self, input_tokens):
# 		""" Embeds both the input tokens and the input labels and adds them """		
# 		# `first_dim` is the batch size multiplied by the number of labels
# 		first_dim, input_length = input_tokens.size()
# 		batch_size 				= int(first_dim/self.num_label_embeddings)

# 		# input_labels = [0, 1, 2]
# 		input_labels = torch.Tensor(range(self.num_label_embeddings)).type_as(input_tokens)
# 		input_labels = input_labels.resize(1, self.num_label_embeddings, 1).repeat(batch_size, 1, input_length)
# 		input_labels = input_labels.resize(first_dim, input_length)
# 		assert input_tokens.size() == input_labels.size()
		
# 		token_embeds = self.embed_tokens(input_tokens)
# 		label_embeds = self.embed_labels(input_labels)
# 		input_embeds = (token_embeds + label_embeds)*.5
# 		return input_embeds