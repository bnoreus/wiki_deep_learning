# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

from pymongo import MongoClient
from bson import ObjectId
import random

# This class retrieves wiki  and builds (summary,query)-pairs.
class DocumentBatcher:
	# Retrieves a set of [size] document summaries and randomly choses a paragraph that is not a summary for one of the documents.
	# Each summary is then paired with that random paragraph and a binary response (Was this the document the paragraph originally came from?)
	def document_pair(self,generator,size):
		documents = []
		for paragraphs in generator():
			if len(paragraphs) > 2:
				documents.append(paragraphs)
			if len(documents) == size:
				# Sample a document to retrieve a query paragraph
				sample_idx = random.randint(0,size-1)
				query_doc = documents[sample_idx]
				query_paragraphs = query_doc[1:] # Remove the summary. We won't sample from that.
				query_paragraph = query_paragraphs[random.randint(0,len(query_paragraphs)-1)]
				
				# Build triplets (summary,query,response)
				for j,d in enumerate(documents):
					response = 1.0 if j==sample_idx else 0.0
					documents[j] = (d[0],query_paragraph,response)
				
				yield documents
				documents = []

	def batch(self,generator,set_size,batch_size):
		a_batch = []
		b_batch = []
		y_batch = []
		for rows in self.document_pair(generator,set_size):
			for a,b,y in rows:
				a_batch.append(a)
				b_batch.append(b)
				y_batch.append(y)
				if len(a_batch) == batch_size:
					yield a_batch,b_batch,y_batch
					a_batch = []
					b_batch = []
					y_batch = []
		yield a_batch,b_batch,y_batch
	def pad_batch(self,text_batch,pad_symbol,text_length=None):
		if text_length is None:
			max_length = 0
			for text in text_batch:
				max_length = max(max_length,len(text))
		else:
			max_length = text_length

		for i,text in enumerate(text_batch):
			if text_length is not None:
				text_batch[i] = text_batch[i][:max_length]
			text_batch[i] += [pad_symbol]*(max_length-len(text))
		return text_batch

	# Creates a iterator over every document used in the training data. This is used when visualing the results.
	def iter_summaries(self,generator):
		for paragraphs in generator():
			if len(paragraphs) > 2:
				summary = paragraphs[0]
				yield "",summary